#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <map>
#include "engine/VulkanEngineOpt.hpp"
#include "engine/RetinaState.hpp"
#include "engine/GemmaVocab.hpp"
#include "utils/HilbertCurve.hpp"

// --- 🚀 真正的 Gemma Tokenizer 轉接器 ---
class GemmaTokenizer {
public:
    GemmaTokenizer() : vocab("weights/gemma4_vocab.bin") {
        std::ifstream f("weights/gemma_embeddings.bin", std::ios::binary);
        if (f) {
            f.seekg(0, std::ios::end);
            size_t size = f.tellg();
            f.seekg(0, std::ios::beg);
            embedding_data = std::make_unique<uint16_t[]>(size / 2);
            f.read(reinterpret_cast<char*>(embedding_data.get()), size);
            std::cout << "💎 Loaded " << size / (1024*1024) << " MB of Embedding weights." << std::endl;
        }
    }

    std::vector<uint32_t> encode(const std::string& text) {
        return vocab.encode(text);
    }

    std::string decode(uint32_t id) {
        auto t = vocab.get_token(id);
        if (t == "[UNK]") return "[" + std::to_string(id) + "]";
        return std::string(t);
    }

    // 獲取 2560 維向量的指標
    const uint16_t* get_embedding(uint32_t id) const {
        if (!embedding_data) return nullptr;
        return &embedding_data[id * 2560];
    }

private:
    GemmaVocab vocab;
    std::unique_ptr<uint16_t[]> embedding_data;
};

int main() {
    std::cout << "===============================================================" << std::endl;
    std::cout << "   CHIMERA-V M10.9: Production-Grade NCA Inference Engine" << std::endl;
    std::cout << "   Arch: Fully-Fused GPU | Topology: 64x64 Hilbert | P: 136 TFLOPS" << std::endl;
    std::cout << "===============================================================" << std::endl;

    using EngineType = VulkanEngineOpt; 

    try {
        EngineType engine;
        GemmaTokenizer tokenizer;
        
        const uint32_t W = 512;
        const uint32_t H = 64;
        const uint32_t C = 2560;
        
        // --- 1. 處理 User Prompt ---
        std::string prompt = "CHIMERA"; // 範例 Prompt
        std::vector<uint32_t> input_ids = tokenizer.encode(prompt);
        std::cout << "[User] Prompt: \"" << prompt << "\" -> Tokens: ";
        for(auto id : input_ids) std::cout << id << " ";
        std::cout << std::endl;

        RetinaState<W, H, C> hostState;
        engine.prepareResources(hostState);
        
        // --- 2. 精準 Embedding 注入 (Hilbert Seeds) ---
        for (size_t i = 0; i < input_ids.size(); ++i) {
            auto pos = HilbertCurve::indexToPoint(64, (uint32_t)i);
            uint16_t* pChannels = hostState.get_pixel_channels(pos.x, pos.y);
            const uint16_t* pEmbed = tokenizer.get_embedding(input_ids[i]);
            if (pEmbed) {
                memcpy(pChannels, pEmbed, C * sizeof(uint16_t));
            }
        }
        
        engine.upload(hostState.data(), hostState.size_bytes());
        engine.upload_persistent_field(hostState.data(), hostState.size_bytes());

        // --- 3. 載入神經權重 ---
        size_t weightBytes = 9ULL * (size_t)C * C * sizeof(uint16_t); 
        engine.loadWeights("weights/gemma_nca_weights_3x3.bin", weightBytes);

        std::cout << "[System] Resources Ready. Starting NCA Latent Evolution..." << std::endl;

        const uint32_t MAX_TOKENS = 12;
        auto start = std::chrono::high_resolution_clock::now();

        // 🚀 執行全融合 GPU 生成 (演化 -> 解碼 -> 注入)
        engine.generate_fused((uint32_t)input_ids.size(), MAX_TOKENS, W, H, C);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        // --- 4. 下載 Token ID 並進行語義映射 ---
        std::vector<uint32_t> token_ids(MAX_TOKENS);
        engine.downloadTokens(token_ids.data(), MAX_TOKENS);

        std::cout << "\n>>> AGENT CONTINUATION SEQUENCE:" << std::endl;
        std::cout << "---------------------------------------------------------------" << std::endl;
        
        for(auto id : token_ids) {
            std::cout << tokenizer.decode(id);
        }
        
        std::cout << "\n---------------------------------------------------------------" << std::endl;

        // --- 效能診斷 ---
        double ops_per_step = (double)W * H * C * C * 9.0 * 2.0 * 42.0; 
        double total_tflops_val = (ops_per_step * MAX_TOKENS) / 1e12;
        double throughput = total_tflops_val / diff.count();

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n[Stats] Throughput: " << throughput << " TFLOPS | Latency: " << (diff.count()/MAX_TOKENS)*1000 << " ms/tok" << std::endl;
        std::cout << "===============================================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "!!! ENGINE ABORT: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
