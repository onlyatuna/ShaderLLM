import torch
import numpy as np
import os

def export_gemma_nca_weights(channels=2560, output_path="weights/gemma_nca_weights.bin"):
    print(f"--- CHIMERA-V Weight Exporter ---")
    print(f"Target Channels: {channels}")
    
    # 建立目錄
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 模擬訓練後的權重 (2560 x 2560)
    # 使用 Xavier Initialization 模擬穩定的初始權重
    std = np.sqrt(2.0 / (channels + channels))
    weights = np.random.randn(channels, channels).astype(np.float16) * std
    
    # 將權重轉為二進位 (Flat binary)
    weights.tofile(output_path)
    
    filesize_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully exported weights to {output_path}")
    print(f"Total Weight Size: {filesize_mb:.2f} MB")

if __name__ == "__main__":
    export_gemma_nca_weights()
