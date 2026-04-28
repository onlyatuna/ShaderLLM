import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# =============================================================================
# CHIMERA-V | Phase 2.1: Fractal Topology (希爾伯特分形拓樸)
# =============================================================================

class HilbertTopology:
    @staticmethod
    def xy2d(n, x, y):
        """將 2D 座標 (x, y) 映射回 1D 距離 d (希爾伯特逆向映射)"""
        d = 0
        s = n // 2
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)
            x, y = HilbertTopology.rot(s, x, y, rx, ry)
            s //= 2
        return d

    @staticmethod
    def d2xy(n, d):
        """將 1D 距離 d 映射回 2D 座標 (x, y) (希爾伯特正向映射)"""
        t = d
        x, y = 0, 0
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = HilbertTopology.rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    @staticmethod
    def rot(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            return y, x
        return x, y

class SpatialCausalMask:
    @staticmethod
    def build(width, height, kernel_size=3):
        n = 1
        while n < max(width, height): n *= 2
        
        mask = torch.ones(kernel_size * kernel_size, height, width)
        pad = kernel_size // 2
        
        print(f"[Mask] Generating Fractal Mask with Hilbert N={n}...")
        
        for y in range(height):
            for x in range(width):
                current_h_idx = HilbertTopology.xy2d(n, x, y)
                k = 0
                for dy in range(-pad, pad + 1):
                    for dx in range(-pad, pad + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            neighbor_h_idx = HilbertTopology.xy2d(n, nx, ny)
                            if neighbor_h_idx > current_h_idx:
                                mask[k, y, x] = 0.0
                        k += 1
        return mask

# =============================================================================
# 其餘代理模型組件與訓練邏輯
# =============================================================================

class ChimeraSurrogateNCA(nn.Module):
    def __init__(self, channels=64, kernel_size=3, width=64, height=64, vocab_size=1000):
        super().__init__()
        self.channels, self.width, self.height, self.kernel_size = channels, width, height, kernel_size
        self.evolve_weight = nn.Parameter(torch.Tensor(channels, channels, kernel_size, kernel_size))
        nn.init.dirac_(self.evolve_weight)
        
        mask = SpatialCausalMask.build(width, height, kernel_size)
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))
        self.lm_head = nn.Linear(channels, vocab_size, bias=False)

    def exact_activation(self, x):
        """
        🚨 破綻三修正：絕對精準的非線性還原 (Exact Activation Matching)
        """
        return x / (1.0 + torch.abs(x))

    def spatial_causal_conv(self, x):
        b, c, h, w = x.shape
        unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size//2)
        unfolded = unfolded.view(b, c, self.kernel_size * self.kernel_size, h, w)
        masked_unfolded = (unfolded * self.causal_mask).view(b, -1, h * w)
        weight_flat = self.evolve_weight.view(self.channels, -1)
        out = torch.matmul(weight_flat, masked_unfolded)
        return out.view(b, self.channels, h, w)

    def forward(self, retina, steps=5):
        x = retina
        for _ in range(steps):
            # 🚨 物理災難二修正：代數衰減的熱寂解藥 (Energy Conservation)
            # 狀態背景保持恆定，非線性激活僅作用於新產生的空間干涉波紋 (變化量 Delta)
            delta = self.spatial_causal_conv(x)
            x = x + self.exact_activation(delta) 
        return x

    def inject_token(self, retina, embedding, token_idx):
        new_retina = retina.clone()
        # 🚨 物理災難一防線：強制確保視網膜為 2^n 正方形以適應希爾伯特拓樸
        assert self.width == self.height and (self.width & (self.width-1) == 0), "CHIMERA-V requires a power-of-2 square retina for Hilbert topology."
        
        n = self.width
        x, y = HilbertTopology.d2xy(n, token_idx)
        
        if y < self.height and x < self.width:
            new_retina[:, :, y, x] = embedding
        return new_retina

    def decode_logits(self, retina, token_idx):
        n = self.width
        x, y = HilbertTopology.d2xy(n, token_idx)
        
        if y < self.height and x < self.width:
            features = retina[:, :, y, x]
        else:
            features = torch.zeros(retina.shape[0], self.channels, device=retina.device)
            
        logits = self.lm_head(features)
        return logits

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🚀 啟動 CHIMERA-V 分形蒸餾與能量守恆擴散 (裝置: {device})...')
    
    channels = 64
    vocab_size = 1000 
    model = ChimeraSurrogateNCA(channels=channels, vocab_size=vocab_size, width=64, height=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size, seq_len = 1, 4
    
    for epoch in range(11):
        opt.zero_grad()
        retina = torch.zeros(batch_size, channels, model.height, model.width, device=device)
        total_loss = 0
        
        for t in range(seq_len):
            in_emb = torch.randn(batch_size, channels, device=device)
            teacher_logits = torch.randn(batch_size, vocab_size, device=device)
            
            retina = model.inject_token(retina, in_emb, t)
            retina_evolved = model.forward(retina, steps=5)
            student_logits = model.decode_logits(retina_evolved, t)
            
            log_student = F.log_softmax(student_logits, dim=-1)
            prob_teacher = F.softmax(teacher_logits, dim=-1)
            loss = F.kl_div(log_student, prob_teacher, reduction='batchmean')
            
            total_loss += loss
            
            # 🚨 物理災難三修正：沿時間反向傳播的記憶體海嘯 (Truncated BPTT)
            # 截斷計算圖，防止梯度無限回溯導致 OOM
            retina = retina_evolved.clone().detach()
        
        total_loss.backward()
        opt.step()
        if epoch % 2 == 0: 
            print(f'Epoch {epoch:02d} | KL Loss: {total_loss.item():.4f}')
    
    os.makedirs('weights', exist_ok=True)
    
    # 🚨 致命破綻二修正：張量步幅物理對齊
    # Vulkan TMA 引擎預期的 3x3 是 [9, OutC, InC] 排列
    weight = model.evolve_weight.data # (OutC, InC, 3, 3)
    weight = weight.permute(2, 3, 0, 1) # (3, 3, OutC, InC)
    weight = weight.reshape(9, model.channels, model.channels) # (9, OutC, InC)
    
    w = weight.cpu().numpy().astype(np.float16)
    w.tofile('weights/gemma_nca_weights_3x3.bin')
    print('✅ 成功匯出無錯位的 9方位 空間卷積權重 至 weights/gemma_nca_weights_3x3.bin')

if __name__ == '__main__':
    train()
