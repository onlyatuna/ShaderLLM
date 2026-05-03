import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import os
import glob
import tqdm

# =============================================================================
# CHIMERA-V | Phase 4.9: Absolute Isomorphism Distiller (M10.9.10)
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
        
        # Base NCA Evolution Kernel
        self.evolve_weight = nn.Parameter(torch.Tensor(channels, channels, kernel_size, kernel_size))
        
        # PLE Injection Components (Simulating Gemma-4 Decoder Layer logic)
        self.ple_gate = nn.Linear(channels, ple_dim)
        self.ple_projection = nn.Linear(ple_dim, channels)
        
        self.eps = 1e-8
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.evolve_weight, mean=0.0, std=0.002)
        with torch.no_grad(): 
            self.evolve_weight[:, :, 1, 1] = -0.05 
        nn.init.xavier_uniform_(self.ple_gate.weight)
        nn.init.zeros_(self.ple_gate.bias)
        nn.init.zeros_(self.ple_projection.weight) # Start with identity-like impact
        nn.init.zeros_(self.ple_projection.bias)

    def thermodynamic_norm(self, x):
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32**2, dim=1, keepdim=True) + self.eps)
        return (x_f32 / rms).type_as(x)

    def evolve_step(self, x, prompt_field, per_layer_ple_signal):
        # 1. Base NCA Step
        x_padded = F.pad(x, pad=(1, 1, 1, 1), mode='circular')
        delta = F.conv2d(x_padded, self.evolve_weight, padding=0)
        x = x + torch.tanh(delta + prompt_field * 0.5)
        
        # 2. PLE Injection (Mathematically equivalent to Gemma-4's per-layer gate)
        # x shape: [1, 2560, 64, 64] -> [1, 4096, 2560]
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Gating
        gate = torch.tanh(self.ple_gate(x_flat)) # Using Tanh for stability in NCA
        
        # Modulation with Layer Signal
        # per_layer_ple_signal shape: [1, 256, 64, 64] -> [4096, 256]
        ple_sig_flat = per_layer_ple_signal.permute(0, 2, 3, 1).reshape(-1, self.ple_dim)
        modulated = gate * ple_sig_flat
        
        # Projection back to hidden space
        ple_out = self.ple_projection(modulated)
        
        # RMSNorm and Residual Add
        rms_ple = torch.sqrt(torch.mean(ple_out**2, dim=-1, keepdim=True) + self.eps)
        ple_out = ple_out / rms_ple
        
        x = x + ple_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.thermodynamic_norm(x)

    def forward(self, x, prompt_field, ple_stack):
        # ple_stack: [42, 1, 256, 64, 64]
        for i in range(42):
            if self.training:
                # 🚀 關鍵最佳化：啟用 Gradient Checkpointing
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

def train_offline_ple():
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    CHANNELS = 2560
    PLE_DIM = 256
    CHUNK_SIZE = 16 # Reduced due to high memory consumption of 42-step grad graph
    
    print(f"🎬 [CHIMERA-V] 啟動 42 步物理同構精煉 (PLE-Enabled | CHUNK: {CHUNK_SIZE})")
    
    # Load PLE Table
    if os.path.exists('weights/gemma_ple_table.bin'):
        print("📂 載入 Gemma-4 全局 PLE 表...")
        ple_table = np.fromfile('weights/gemma_ple_table.bin', dtype=np.float16).reshape(262144, 42, 256)
        # 🚀 關鍵最佳化：把 11GB 的表留在 CPU！
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
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    cos_loss_fn = nn.CosineEmbeddingLoss(reduction='none')

    for epoch in range(30):
        total_loss = 0
        recal_interval = 8 # Constant for stability
        
        for data in dataset:
            input_ids = data['input_ids'][0].to(device) # [seq_len]
            in_feat = data['input'].to(device)
            target_feat = data['target'].to(device)
            seq_len = in_feat.shape[1]

            retina = torch.zeros(1, CHANNELS, 64, 64, device=device, requires_grad=True)
            prompt_field = torch.zeros(1, CHANNELS, 64, 64, device=device)
            # PLE State: [42, 1, 256, 64, 64]
            ple_stack = torch.zeros(42, 1, PLE_DIM, 64, 64, device=device)
            
            # Init first token
            tx0, ty0 = d_to_xy[0]
            with torch.no_grad():
                retina[0, :, ty0, tx0] = in_feat[0, 0]
                prompt_field[0, :, ty0, tx0] = in_feat[0, 0]
                # Inject initial PLE signal
                token_id = input_ids[0].item() # 轉成 int 取 CPU tensor
                # 🚀 從 CPU 抽出現有 Token 的 PLE 再丟到 GPU
                ple_stack[:, 0, :, ty0, tx0] = ple_table[token_id].to(device) 
            
            optimizer.zero_grad()
            chunk_loss = 0.0
            
            for t in range(seq_len - 1):
                tx, ty = d_to_xy[t+1]
                
                with torch.amp.autocast('cuda'):
                    # 42 Steps of Evolution with PLE
                    retina = student(retina, prompt_field, ple_stack)
                    
                    step_loss = get_halo_loss(retina, target_feat[0, t+1], tx, ty, t+1, xy_to_d, cos_loss_fn,
                                             pre_off_y, pre_off_x, pre_dist_w, pre_ones_label)
                    chunk_loss = chunk_loss + step_loss
                
                if (t + 1) % CHUNK_SIZE == 0 or (t + 1) == (seq_len - 1):
                    scaler.scale(chunk_loss).backward()
                    total_loss += chunk_loss.item()
                    chunk_loss = 0.0
                    retina = retina.detach().requires_grad_(True)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

                if (t + 1) % recal_interval == 0:
                    retina = retina.clone()
                    retina[:, :, ty, tx] = in_feat[0, t + 1]
                    prompt_field = prompt_field.clone()
                    prompt_field[:, :, ty, tx] = in_feat[0, t + 1]
                    # Update PLE stack for the new token
                    token_id = input_ids[t+1].item()
                    ple_stack = ple_stack.clone()
                    ple_stack[:, 0, :, ty, tx] = ple_table[token_id].to(device)

        print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")

    # Export Weights
    os.makedirs('weights', exist_ok=True)
    
    # 1. NCA Weights [out, ky, kx, in]
    w = student.evolve_weight.data.cpu().numpy().astype(np.float16)
    w_trans = np.transpose(w, (0, 2, 3, 1))
    w_trans.reshape(CHANNELS, 9 * CHANNELS).tofile('weights/gemma_nca_weights_3x3.bin')
    
    # 2. PLE Weights
    student.ple_gate.weight.data.cpu().numpy().astype(np.float16).tofile('weights/gemma_ple_gate_w.bin')
    student.ple_gate.bias.data.cpu().numpy().astype(np.float16).tofile('weights/gemma_ple_gate_b.bin')
    student.ple_projection.weight.data.cpu().numpy().astype(np.float16).tofile('weights/gemma_ple_proj_w.bin')
    student.ple_projection.bias.data.cpu().numpy().astype(np.float16).tofile('weights/gemma_ple_proj_b.bin')
    
    print("✅ 物理同構精煉完成！所有對齊權重已匯出至 weights/ 目錄。")

if __name__ == "__main__":
    train_offline_ple()

