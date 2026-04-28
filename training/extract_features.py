import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import tqdm

# =============================================================================
# CHIMERA-V | Phase 4.7.2: PLE-Aware Feature Extraction
# =============================================================================

def extract_gemma_features():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "google/gemma-4-E4B" # 假設此為目標模型路徑
    output_dir = "data/features"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎬 [Stage 1] 正在從 {model_id} 挖掘語義特徵與 PLE 錨點...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    prompts = [
        "The laws of physics govern the universe.",
        "Artificial intelligence is changing the world.",
        "In the heart of the city, technology meets nature.",
        "The neural cellular automata simulates biological growth.",
        "High-performance computing requires low-level optimization.",
        "Quantum mechanics describes the subatomic scale.",
        "The chimera of mythology is a multi-headed beast.",
        "Data structures and algorithms are the foundation of software.",
        "A quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "The future belongs to those who believe in the beauty of their dreams.",
        "The only limit to our realization of tomorrow is our doubts of today."
    ]

    for i, text in enumerate(tqdm.tqdm(prompts)):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids.detach().cpu()
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # 🚀 起點：Layer 0 (Embedding Output)
            in_feat = outputs.hidden_states[0].detach().cpu()
            
            # 🚀 終點：Final RMSNorm Output
            final_hidden = outputs.hidden_states[-1]
            out_feat = model.model.language_model.norm(final_hidden).detach().cpu()
            
            torch.save({
                'input_ids': input_ids, # [1, seq_len]
                'input': in_feat,      # [1, seq_len, 2560]
                'target': out_feat     # [1, seq_len, 2560]
            }, f"{output_dir}/feat_{i:03d}.pt")
            
            del outputs
            torch.cuda.empty_cache()

    print(f"✅ PLE 特徵挖掘完成！數據已存入 {output_dir}")

if __name__ == "__main__":
    extract_gemma_features()
