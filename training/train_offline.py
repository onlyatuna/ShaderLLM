import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import os
import glob
import tqdm

# =============================================================================
# CHIMERA-V | Phase 4.9.1: Absolute Isomorphism Distiller (Perfect Alignment)
# =============================================================================

class HilbertTopology:
    @staticmethod
    def get_luts(n, device):
        d_to_xy = torch.zeros((n*n, 2), dtype=torch.long, device=device)
        xy_to_d = torch.zeros((n, n), dtype=torch.long, device=device)
        for d in range(n*n):
            t = d; x, y = 0, 0; s = 1
            while s < n:
                rx = 1 & (t // 2); ry = 1 & (t ^ rx)
                if ry == 0:
                    if rx == 1: x = s - 1 - x; y = s - 1 - y
                    x, y = y, x
                x += s * rx; y += s * ry; t //= 4; s *= 2
            d_to_xy[d] = torch.tensor([x, y]); xy_to_d[y, x] = d
        return d_to_xy, xy_to_d

class ChimeraNCA_PLE(nn.Module):
    def __init__(self, channels=2560, ple_dim=256, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.ple_dim = ple_dim
        
        self.evolve_weight = nn.Parameter(torch.Tensor(channels, channels, kernel_size, kernel_size))
        self.ple_gate = nn.Linear(channels, ple_dim, bias=False) # Shader 裡沒有 bias
        self.ple_projection = nn.Linear(ple_dim, channels, bias=False) # Shader 裡沒有 bias
        
        self.eps = 1e-8
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.evolve_weight, mean=0.0, std=0.002)
        with torch.no_grad(): 
            self.evolve_weight[:, :, 1, 1] = -0.05 
        nn.init.xavier_uniform_(self.ple_gate.weight)
        nn.init.zeros_(self.ple_projection.weight)

    def thermodynamic_norm(self, x):
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32**2, dim=1, keepdim=True) + self.eps)
        return (x_f32 / rms).type_as(x)

    def evolve_step(self, x, prompt_field, per_layer_ple_signal):
        B, C, H, W = x.shape
        
        # 🚀 階段 1：PLE 閘門注入 (嚴格對齊 nca_ple_gate.slang)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        gate = self.ple_gate(x_flat) # 不加 tanh，對齊 shader 的 sum
        ple_sig_flat = per_layer_ple_signal.permute(0, 2, 3, 1).reshape(-1, self.ple_dim)
        modulated = gate * ple_sig_flat
        ple_out = self.ple_projection(modulated)
        
        # Shader 中的 Tanh 避震器：retinaState[targetIdx] += half(tanh(sum) * 0.5f);
        safe_ple_out = torch.tanh(ple_out) * 0.5
        x_ple = x + safe_ple_out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 🚀 階段 2：NCA 空間演化 (嚴格對齊 nca_spatial_agg + nca_evolve)
        x_padded = F.pad(x_ple, pad=(1, 1, 1, 1), mode='circular')
        delta = F.conv2d(x_padded, self.evolve_weight, padding=0)
        x_evolved = x_ple + torch.tanh(delta + prompt_field * 0.5)
        
        # 🚀 階段 3：全局 RMSNorm (嚴格對齊 nca_rmsnorm)
        return self.thermodynamic_norm(x_evolved)

    def forward(self, x, prompt_field, ple_stack):
        for i in range(42):
            if self.training:
                x = checkpoint(self.evolve_step, x, prompt_field, ple_stack[i], use_reentrant=False)
            else:
                x = self.evolve_step(x, prompt_field, ple_stack[i])
        return x

def get_halo_loss(student_retina, target_feature, tx, ty, current_t, xy_to_d, cos_loss_fn, 
                  pre_off_y, pre_off_x, pre_dist_w, pre_ones_label):
    nx = (tx + pre_off_x) % 64; ny = (ty + pre_off_y) % 64
    mask = (xy_to_d[ny, nx] <= current_t).float()
    final_weight = (mask * pre_dist_w).view(-1, 1)
    pixels = student_retina[0, :, ny, nx].permute(1, 2, 0).reshape(9, -1)
    target = target_feature.unsqueeze(0).expand(9, -1)
    mse = F.mse_loss(pixels, target, reduction='none').mean(dim=1, keepdim=True)
    cosine = cos_loss_fn(pixels, target, pre_ones_label).unsqueeze(1)
    return ((0.3 * mse + 0.7 * cosine) * final_weight).sum() / mask.sum().clamp(min=1)

def load_checkpoint_if_exists(student, channels, ple_dim, device):
    weight_path = 'weights/gemma_nca_weights_3x3.bin'
    if os.path.exists(weight_path):
        print("🔄 偵測到現有權重，啟動中斷點接續 (Resume from Checkpoint)...")
        try:
            # Load NCA weights
            w_trans = np.fromfile(weight_path, dtype=np.float16).reshape(channels, 9 * channels)
            w = np.transpose(w_trans.reshape(channels, 3, 3, channels), (0, 3, 1, 2))
            student.evolve_weight.data = torch.from_numpy(w.astype(np.float32)).to(device)
            
            # Load PLE weights
            gate_w = np.fromfile('weights/gemma_ple_gate_w.bin', dtype=np.float16).reshape(ple_dim, channels)
            student.ple_gate.weight.data = torch.from_numpy(gate_w.astype(np.float32)).to(device)
            proj_w = np.fromfile('weights/gemma_ple_proj_w.bin', dtype=np.float16).reshape(channels, ple_dim)
            student.ple_projection.weight.data = torch.from_numpy(proj_w.astype(np.float32)).to(device)
            print("✅ 成功載入歷史神經節點！")
        except Exception as e:
            print(f"⚠️ 讀取歷史權重失敗，將從頭開始訓練: {e}")

def train_offline_ple():
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    CHANNELS = 2560
    PLE_DIM = 256
    CHUNK_SIZE = 16 
    
    print(f"🎬 [CHIMERA-V] 啟動 42 步物理同構精煉 (Perfect Alignment | CHUNK: {CHUNK_SIZE})")
    
    if os.path.exists('weights/gemma_ple_table.bin'):
        print("📂 載入 Gemma-4 全局 PLE 表 (置於 CPU 以防 OOM)...")
        ple_table = np.fromfile('weights/gemma_ple_table.bin', dtype=np.float16).reshape(262144, 42, 256)
        ple_table = torch.from_numpy(ple_table.astype(np.float32)) 
    else:
        print("⚠️ 找不到 weights/gemma_ple_table.bin，將使用虛擬 PLE 表訓練。")
        ple_table = torch.randn(262144, 42, 256) * 0.01

    d_to_xy, xy_to_d = HilbertTopology.get_luts(64, device)
    offsets = torch.tensor([-1, 0, 1], device=device)
    pre_off_y, pre_off_x = torch.meshgrid(offsets, offsets, indexing='ij')
    pre_dist_w = torch.ones((3, 3), device=device); pre_dist_w[torch.max(torch.abs(pre_off_x), torch.abs(pre_off_y)) == 1] = 0.5
    pre_ones_label = torch.ones(9, device=device)

    feature_files = glob.glob("data/features/*.pt")
    if not feature_files:
        print("❌ 找不到離線特徵數據！請先執行 extract_features.py")
        return
    
    dataset = []
    for fpath in feature_files:
        dataset.append(torch.load(fpath, weights_only=True))

    student = ChimeraNCA_PLE(channels=CHANNELS, ple_dim=PLE_DIM).to(device)
    
    # 🚀 自動接續功能
    load_checkpoint_if_exists(student, CHANNELS, PLE_DIM, device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-6)
    cos_loss_fn = nn.CosineEmbeddingLoss(reduction='none')

    for epoch in range(30):
        total_loss = 0
        recal_interval = 8
        
        # 打亂資料集增加泛化性
        np.random.shuffle(dataset) 
        
        for data in dataset:
            input_ids = data['input_ids'][0] 
            # 🚀 強制轉型為 FP32，匹配神經網路純單精度運算
            in_feat = data['input'].to(device).float()
            target_feat = data['target'].to(device).float()
            seq_len = in_feat.shape[1]

            retina = torch.zeros(1, CHANNELS, 64, 64, device=device, requires_grad=True)
            prompt_field = torch.zeros(1, CHANNELS, 64, 64, device=device)
            ple_stack = torch.zeros(42, 1, PLE_DIM, 64, 64, device=device)
            
            tx0, ty0 = d_to_xy[0]
            with torch.no_grad():
                retina[0, :, ty0, tx0] = in_feat[0, 0]
                prompt_field[0, :, ty0, tx0] = in_feat[0, 0]
                token_id = input_ids[0].item()
                ple_stack[:, 0, :, ty0, tx0] = ple_table[token_id].to(device) 
            
            optimizer.zero_grad()
            chunk_loss = 0.0
            
            for t in range(seq_len - 1):
                tx, ty = d_to_xy[t+1]
                
                # 純 FP32 穩定訓練
                retina = student(retina, prompt_field, ple_stack)
                
                step_loss = get_halo_loss(retina, target_feat[0, t+1], tx, ty, t+1, xy_to_d, cos_loss_fn,
                                         pre_off_y, pre_off_x, pre_dist_w, pre_ones_label)
                chunk_loss = chunk_loss + step_loss
                
                if (t + 1) % CHUNK_SIZE == 0 or (t + 1) == (seq_len - 1):
                    chunk_loss.backward()
                    total_loss += chunk_loss.item()
                    chunk_loss = 0.0
                    retina = retina.detach().requires_grad_(True)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad()

                if (t + 1) % recal_interval == 0:
                    retina = retina.clone()
                    retina[:, :, ty, tx] = in_feat[0, t + 1]
                    prompt_field = prompt_field.clone()
                    prompt_field[:, :, ty, tx] = in_feat[0, t + 1]
                    token_id = input_ids[t+1].item()
                    ple_stack = ple_stack.clone()
                    ple_stack[:, 0, :, ty, tx] = ple_table[token_id].to(device)

        print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")

        os.makedirs('weights', exist_ok=True)
        with torch.no_grad():
            w = student.evolve_weight.data.cpu().numpy().astype(np.float16)
            w_trans = np.transpose(w, (0, 2, 3, 1))
            w_trans.reshape(CHANNELS, 9 * CHANNELS).tofile('weights/gemma_nca_weights_3x3.bin')
            student.ple_gate.weight.data.cpu().numpy().astype(np.float16).tofile('weights/gemma_ple_gate_w.bin')
            student.ple_projection.weight.data.cpu().numpy().astype(np.float16).tofile('weights/gemma_ple_proj_w.bin')
            # 儲存空 bias 以相容 C++ 讀取邏輯
            np.zeros(PLE_DIM, dtype=np.float16).tofile('weights/gemma_ple_gate_b.bin')
            np.zeros(CHANNELS, dtype=np.float16).tofile('weights/gemma_ple_proj_b.bin')
            
        print(f"💾 Epoch {epoch:02d} 權重已自動同步至 weights/")

if __name__ == "__main__":
    train_offline_ple()