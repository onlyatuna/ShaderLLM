import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM

def export_ple_weights():
    model_id = "google/gemma-4-E4B"
    print(f"🎬 準備萃取 {model_id} 的 PLE 全域發射器與動態閘門...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu", 
        trust_remote_code=True
    )

    os.makedirs("weights", exist_ok=True)
    
    print("1. 匯出全域 PLE 表 (Global PLE Table) ...")
    # 取得 262144 x 10752 的超大矩陣 (約 5.6 GB)
    ple_table = model.model.language_model.embed_tokens_per_layer.weight.data
    ple_table.cpu().numpy().astype(np.float16).tofile("weights/gemma_ple_table.bin")
    print(f"   -> gemma_ple_table.bin 匯出完成！大小: {os.path.getsize('weights/gemma_ple_table.bin') / (1024**3):.2f} GB")

    print("2. 匯出 42 層的動態閘門權重 (Layer Gates & Projections) ...")
    # 將 42 層的 gate_weight (2560->256) 與 proj_weight (256->2560) 壓扁連續儲存
    # 以及 RMSNorm 的 weight
    ple_weights = []
    for i in range(42):
        layer = model.model.language_model.layers[i]
        # W_gate: (256, 2560)
        ple_weights.append(layer.per_layer_input_gate.weight.data.cpu().numpy().astype(np.float16).flatten())
        # W_proj: (2560, 256)
        ple_weights.append(layer.per_layer_projection.weight.data.cpu().numpy().astype(np.float16).flatten())
        # Norm weight: (2560)
        ple_weights.append(layer.post_per_layer_input_norm.weight.data.cpu().numpy().astype(np.float16).flatten())

    concatenated_weights = np.concatenate(ple_weights)
    concatenated_weights.tofile("weights/gemma_ple_weights.bin")
    print(f"   -> gemma_ple_weights.bin 匯出完成！大小: {os.path.getsize('weights/gemma_ple_weights.bin') / (1024**2):.2f} MB")
    
    print("✅ PLE 系統萃取完成！")

if __name__ == "__main__":
    export_ple_weights()
