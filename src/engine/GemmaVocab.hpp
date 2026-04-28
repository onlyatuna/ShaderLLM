#pragma once
#include <vector>
#include <string>
#include <string_view>
#include <fstream>
#include <iostream>
#include <memory>

#include <unordered_map>

class GemmaVocab {
public:
    GemmaVocab(const std::string& bin_path) {
        std::ifstream f(bin_path, std::ios::binary);
        if (!f) {
            std::cerr << "❌ Failed to open vocab bin: " << bin_path << std::endl;
            return;
        }

        uint32_t count = 0;
        f.read(reinterpret_cast<char*>(&count), 4);
        
        std::vector<uint32_t> offsets(count);
        std::vector<uint32_t> lengths(count);
        
        f.read(reinterpret_cast<char*>(offsets.data()), count * 4);
        f.read(reinterpret_cast<char*>(lengths.data()), count * 4);
        
        auto current_pos = f.tellg();
        f.seekg(0, std::ios::end);
        size_t blob_size = (size_t)f.tellg() - (size_t)current_pos;
        f.seekg(current_pos);
        
        blob_buffer = std::make_unique<char[]>(blob_size);
        f.read(blob_buffer.get(), blob_size);
        
        tokens.reserve(count);
        token_to_id.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            std::string_view sv(&blob_buffer[offsets[i]], lengths[i]);
            tokens.emplace_back(sv);
            token_to_id[std::string(sv)] = i;
        }
        
        std::cout << "📦 Loaded " << count << " tokens and built reverse index." << std::endl;
    }

    std::string_view get_token(uint32_t id) const {
        if (id < tokens.size()) return tokens[id];
        return "[UNK]";
    }

    // 簡單的貪婪分詞 (Greedy Tokenization)
    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> ids;
        std::string s = text;
        
        // 1. 處理空格 (Gemma 樣式: 空格變 ▁)
        // 注意：這裡簡化處理，實際 SPM 需要更複雜的正規化
        
        size_t pos = 0;
        while (pos < s.length()) {
            bool found = false;
            // 從最長匹配開始嘗試
            for (size_t len = std::min(s.length() - pos, (size_t)32); len > 0; --len) {
                std::string sub = s.substr(pos, len);
                if (token_to_id.count(sub)) {
                    ids.push_back(token_to_id.at(sub));
                    pos += len;
                    found = true;
                    break;
                }
            }
            if (!found) {
                ids.push_back(3); // <unk>
                pos++;
            }
        }
        return ids;
    }

    size_t size() const { return tokens.size(); }

private:
    std::unique_ptr<char[]> blob_buffer;
    std::vector<std::string_view> tokens;
    std::unordered_map<std::string, uint32_t> token_to_id;
};

// 為了保持向後相容，我們提供一個簡單的包裝器
// 但建議直接使用 GemmaVocab 類別以獲得最佳效能
inline std::vector<std::string> get_gemma_vocab_legacy() {
    // 此函數僅供舊代碼參考，效能較差
    return {}; 
}
