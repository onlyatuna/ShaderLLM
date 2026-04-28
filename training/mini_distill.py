import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class ChimeraSurrogateNCA(nn.Module):
    def __init__(self, channels=64, kernel_size=3, width=512, height=64):
        super().__init__()
        self.channels, self.width, self.height = channels, width, height
        self.evolve_kernel = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=False)
        nn.init.dirac_(self.evolve_kernel.weight)
    def forward(self, retina, steps=5):
        # 避免 In-place 修改，確保 Autograd 追蹤
        x = retina
        for _ in range(steps):
            # 不再使用 x = ...，確保每一步都是新張量
            x = torch.tanh(self.evolve_kernel(x))
        return x
    def inject_token(self, retina, embedding, token_idx):
        # 使用 clone() 避免修改原始緩衝區
        new_retina = retina.clone()
        y = (token_idx // self.width) % self.height
        x = token_idx % self.width
        # 這裡的賦值要小心
        new_retina[0, :, y, x] = embedding[0]
        return new_retina
    def decode_token_energy(self, retina):
        energy = torch.sum(torch.abs(retina[:, :8, :, :]), dim=1)
        # 提取特徵，不修改 retina
        max_idx = torch.argmax(energy[0].view(-1))
        y, x = max_idx // self.width, max_idx % self.width
        return retina[0, :, y, x].unsqueeze(0)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training M9-C on {device}...')
    model = ChimeraSurrogateNCA(channels=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(11):
        opt.zero_grad()
        # 初始化
        retina = torch.zeros(1, 64, 64, 512, device=device)
        total_loss = 0
        for t in range(2): # 跑 2 個 Token 即可驗證
            in_emb = torch.randn(1, 64, device=device)
            target = torch.randn(1, 64, device=device)
            retina = model.inject_token(retina, in_emb, t)
            retina = model.forward(retina, steps=3) # 減少步數
            out = model.decode_token_energy(retina)
            total_loss += F.mse_loss(out, target)
        
        total_loss.backward()
        opt.step()
        if epoch % 2 == 0: print(f'Epoch {epoch} | Loss: {total_loss.item():.4f}')
    
    os.makedirs('weights', exist_ok=True)
    w = model.evolve_kernel.weight.data[:, :, 1, 1].cpu().numpy().T.astype(np.float16)
    w.tofile('weights/gemma_nca_weights.bin')
    print('✅ Exported gemma_nca_weights.bin (256 KB for 64-channels)')

if __name__ == '__main__':
    train()
