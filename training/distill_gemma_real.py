import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
from torch.utils.checkpoint import checkpoint

# =============================================================================
# CHIMERA-V | Phase 4.6: VRAM-Optimized Industrial Distiller (M10.9.7)
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

class ChimeraNCA(nn.Module):
    def __init__(self, channels=2560, kernel_size=3, width=64, height=64):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.evolve_weight = nn.Parameter(torch.Tensor(channels, channels, kernel_size, kernel_size))
        nn.init.normal_(self.evolve_weight, mean=0.0, std=0.002)
        with torch.no_grad(): self.evolve_weight[:, :, 1, 1] = -0.05 
        self.eps = 1e-8

    def thermodynamic_norm(self, x):
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32**2, dim=1, keepdim=True) + self.eps)
        return (x_f32 / rms).type_as(x)

    def evolve_step(self, x, prompt_field):
        x_padded = F.pad(x, pad=(1, 1, 1, 1), mode='circular')
        delta = F.conv2d(x_padded, self.evolve_weight, padding=0)
        x = x + torch.tanh(delta + prompt_field * 0.5)
        return self.thermodynamic_norm(x)

    def forward(self, x, prompt_field, steps=3, use_checkpoint=False):
        for _ in range(steps):
            if use_checkpoint and self.training:
                x = checkpoint(self.evolve_step, x, prompt_field, use_reentrant=False)
            else:
                x = self.evolve_step(x, prompt_field)
        return x

def get_halo_loss_zero_alloc(student_retina, target_feature, tx, ty, current_t, xy_to_d, cos_loss_fn, 
                             precalc_offsets_y, precalc_offsets_x, precalc_dist_weights, precalc_ones_label):
    nx = (tx + precalc_offsets_x) % 64; ny = (ty + precalc_offsets_y) % 64
    mask = (xy_to_d[ny, nx] <= current_t).float()
    final_weight = (mask * precalc_dist_weights).view(-1, 1)
    pixels = student_retina[0, :, ny, nx].permute(1, 2, 0).reshape(9, -1)
    target = target_feature.unsqueeze(0).expand(9, -1)
    mse = F.mse_loss(pixels, target, reduction='none').mean(dim=1, keepdim=True)
    cosine = cos_loss_fn(pixels, target, precalc_ones_label).unsqueeze(1)
    return ((0.3 * mse + 0.7 * cosine) * final_weight).sum() / mask.sum().clamp(min=1)

def train_real_distill():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "google/gemma-4-E4B" 
    CHANNELS = 2560
    
    print(f"🎬 啟動黑洞封印版精煉 (裝置: {device})")
    
    d_to_xy, xy_to_d = HilbertTopology.get_luts(64, device)
    offsets = torch.tensor([-1, 0, 1], device=device)
    pre_off_y, pre_off_x = torch.meshgrid(offsets, offsets, indexing='ij')
    dist = torch.max(torch.abs(pre_off_x), torch.abs(pre_off_y))
    pre_dist_w = torch.ones_like(dist, dtype=torch.float32); pre_dist_w[dist == 1] = 0.5
    pre_ones_label = torch.ones(9, device=device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    teacher = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    student = ChimeraNCA(channels=CHANNELS).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler('cuda')
    cos_loss_fn = nn.CosineEmbeddingLoss(reduction='none')

    prompts = ["The laws of physics govern the universe.", "Artificial intelligence is changing the world."]

    for epoch in range(11):
        recalibration_interval = max(1, epoch // 2 + 1) 
        total_loss_val = 0
        for text in prompts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = teacher(**inputs, output_hidden_states=True)
                teacher_layer_in = outputs.hidden_states[2].detach().clone()
                teacher_layer_out = outputs.hidden_states[4].detach().clone()
                del outputs
                torch.cuda.empty_cache()

            seq_len = teacher_layer_in.shape[1]
            retina = torch.zeros(1, CHANNELS, 64, 64, device=device, requires_grad=True)
            prompt_field = torch.zeros(1, CHANNELS, 64, 64, device=device)
            
            tx0, ty0 = d_to_xy[0]
            with torch.no_grad():
                retina[0, :, ty0, tx0] = teacher_layer_in[0, 0]
                prompt_field[0, :, ty0, tx0] = teacher_layer_in[0, 0]
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                for t in range(seq_len - 1):
                    tx, ty = d_to_xy[t+1]
                    retina = student(retina, prompt_field, steps=3, use_checkpoint=(seq_len > 16))
                    loss = get_halo_loss_zero_alloc(retina, teacher_layer_out[0, t+1], tx, ty, t+1, xy_to_d, cos_loss_fn,
                                                   pre_off_y, pre_off_x, pre_dist_w, pre_ones_label)
                    scaler.scale(loss).backward()
                    total_loss_val += loss.item()
                    retina = retina.detach().requires_grad_(True) 

                    if (t + 1) % recalibration_interval == 0:
                        # 🚨 關鍵修正：建立新節點以避開 In-place 錯誤
                        retina_new = retina.clone()
                        retina_new[:, :, ty, tx] = teacher_layer_in[0, t + 1]
                        retina = retina_new
                        
                        prompt_field_new = prompt_field.clone()
                        prompt_field_new[:, :, ty, tx] = teacher_layer_in[0, t + 1]
                        prompt_field = prompt_field_new

            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

        print(f"Epoch {epoch:02d} | Loss: {total_loss_val:.6f} | Calibrate: {recalibration_interval}")

    os.makedirs('weights', exist_ok=True)
    w = student.evolve_weight.data.cpu().numpy().astype(np.float16)
    np.transpose(w, (2, 3, 0, 1)).reshape(9, CHANNELS, CHANNELS).tofile('weights/gemma_nca_weights_3x3.bin')
    print("✅ 黑洞封印精煉完成！")

if __name__ == "__main__":
    train_real_distill()
