#include "llme_service.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <ggml.h>
#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>

using json = nlohmann::json;

LlmeService::LlmeService() {
    status_ = "서비스 초기화 중";
    loadModelsFromJson();
    status_ = "로드된 모델 없음";
}

void LlmeService::loadModelsFromJson() {
    std::lock_guard<std::mutex> lock(mtx_);
    available_models_.clear();
    std::ifstream in("/tmp/models.json");
    if (in) {
        try {
            json j;
            in >> j;
            for (const auto& item : j) {
                ModelInfo m;
                m.type = item.value("type", "");
                m.name = item.value("name", "");
                m.model = item.value("model", "");
                m.mmproj = item.value("mmproj", "");
                m.model_id = "";
                available_models_.push_back(m);
            }
        } catch (...) {}
    }
}

std::string LlmeService::getStatus() {
    std::lock_guard<std::mutex> lock(mtx_);
    return status_;
}

std::vector<json> LlmeService::getModels() {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<json> result;
    for (const auto& m : available_models_) {
        json j = {
            {"type", m.type},
            {"name", m.name},
            {"model", m.model}
        };
        if (!m.mmproj.empty()) j["mmproj"] = m.mmproj;
        result.push_back(j);
    }
    return result;
}

bool LlmeService::cancel() {
    std::lock_guard<std::mutex> lock(mtx_);
    // 실제 모델 실행 취소 로직 필요
    status_ = "모델 실행 취소됨";
    return true;
}

std::string LlmeService::generateModelId(const std::string& name) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::stringstream ss;
    ss << name << "-" << ms;
    return ss.str();
}

json LlmeService::load(const std::string& name) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& m : available_models_) {
        if (m.name == name) {
            status_ = "모델 로드 중";
            m.model_id = generateModelId(name);
            running_models_[m.model_id] = m;
            current_model_ = m;
            status_ = "모델 로드 완료";
            json j = {
                {"model_id", m.model_id},
                {"type", m.type},
                {"name", m.name},
                {"model", m.model}
            };
            if (!m.mmproj.empty()) j["mmproj"] = m.mmproj;
            return j;
        }
    }
    return json();
}

bool LlmeService::unload(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = running_models_.find(model_id);
    if (it != running_models_.end()) {
        status_ = "모델 언로드 중";
        running_models_.erase(it);
        if (current_model_ && current_model_->model_id == model_id) current_model_.reset();
        status_ = "모델 언로드 완료";
        return true;
    }
    return false;
}

std::vector<json> LlmeService::getRunningModel() {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<json> result;
    for (const auto& [id, m] : running_models_) {
        json j = {
            {"model_id", m.model_id},
            {"type", m.type},
            {"name", m.name},
            {"model", m.model}
        };
        if (!m.mmproj.empty()) j["mmproj"] = m.mmproj;
        result.push_back(j);
    }
    return result;
}

std::vector<float> LlmeService::embed(const std::string& model_id, const std::string& text) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = running_models_.find(model_id);
    if (it == running_models_.end() || it->second.type != "embedding") return {};
    const auto& m = it->second;

    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    llama_model* model = llama_model_load_from_file(m.model.c_str(), model_params);
    if (!model) return {};
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { llama_model_free(model); return {}; }

    // 토크나이즈
    int max_tokens = 512;
    std::vector<llama_token> tokens(max_tokens);
    int n_tokens = llama_tokenize(llama_model_get_vocab(model), text.c_str(), text.size(), tokens.data(), max_tokens, true, false);
    if (n_tokens < 0) { llama_free(ctx); llama_model_free(model); return {}; }
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    int rc = llama_decode(ctx, batch);
    if (rc != 0) { llama_free(ctx); llama_model_free(model); return {}; }

    int n_embd = llama_model_n_embd(model);
    float* emb = llama_get_embeddings(ctx);
    std::vector<float> result(emb, emb + n_embd);

    llama_free(ctx);
    llama_model_free(model);
    return result;
}

std::string LlmeService::query(const std::string& model_id, const std::string& text) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = running_models_.find(model_id);
    if (it == running_models_.end() || it->second.type != "llm") return "";
    const auto& m = it->second;

    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();
    llama_model* model = llama_model_load_from_file(m.model.c_str(), model_params);
    if (!model) return "";
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { llama_model_free(model); return ""; }

    int max_tokens = 512;
    std::vector<llama_token> tokens(max_tokens);
    int n_tokens = llama_tokenize(llama_model_get_vocab(model), text.c_str(), text.size(), tokens.data(), max_tokens, true, false);
    if (n_tokens < 0) { llama_free(ctx); llama_model_free(model); return ""; }
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    int rc = llama_decode(ctx, batch);
    if (rc != 0) { llama_free(ctx); llama_model_free(model); return ""; }

    // 샘플링: greedy
    std::string result;
    auto* vocab = llama_model_get_vocab(model);
    for (int i = 0; i < 64; ++i) {
        float* logits = llama_get_logits(ctx);
        if (!logits) break;
        
        int n_vocab = llama_vocab_n_tokens(vocab);
        int max_id = 0;
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; ++j) {
            if (logits[j] > max_logit) { max_logit = logits[j]; max_id = j; }
        }
        
        llama_token eos_token = llama_vocab_eos(vocab);
        if (max_id == eos_token) break;
        
        char buf[32] = {0};
        int n_chars = llama_token_to_piece(vocab, max_id, buf, sizeof(buf) - 1, 0, false);
        if (n_chars > 0) {
            result += std::string(buf, n_chars);
        }
        
        llama_token next_token = max_id;
        llama_batch next_batch = llama_batch_get_one(&next_token, 1);
        llama_decode(ctx, next_batch);
    }
    llama_free(ctx);
    llama_model_free(model);
    return result;
}

std::string LlmeService::queryImage(const std::string& model_id, const std::string& text, const std::string& image_path) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = running_models_.find(model_id);
    if (it == running_models_.end() || it->second.type != "multimodal") return "";
    const auto& m = it->second;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99; // GPU 레이어 수 설정 (모든 레이어를 GPU로)
    llama_model* model = llama_model_load_from_file(m.model.c_str(), model_params);
    if (!model) return "";
    mtmd_context_params ctx_params = mtmd_context_params_default();
    mtmd_context* mmctx = mtmd_init_from_file(m.mmproj.c_str(), model, ctx_params);
    if (!mmctx) { llama_model_free(model); return ""; }

    mtmd_bitmap* bitmap = mtmd_helper_bitmap_init_from_file(mmctx, image_path.c_str());
    if (!bitmap) { mtmd_free(mmctx); llama_model_free(model); return ""; }

    // Qwen-VL에 맞는 챗 템플릿 형식 사용 
    std::string prompt_with_marker = "<|im_start|>user\n<__media__>" + text + "<|im_end|>\n<|im_start|>assistant\n";
    mtmd_input_text input_text = { prompt_with_marker.c_str(), true, true };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[1] = { bitmap };
    if (mtmd_tokenize(mmctx, chunks, &input_text, bitmaps, 1) != 0) {
        mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return "";
    }

    llama_context_params lctx_params = llama_context_default_params();
    lctx_params.n_ctx = 4096; // 컨텍스트 크기 설정
    lctx_params.n_batch = 512; // 배치 크기 설정
    llama_context* lctx = llama_init_from_model(model, lctx_params);
    if (!lctx) { mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return ""; }

    llama_pos n_past = 0;
    int32_t rc = mtmd_helper_eval_chunks(mmctx, lctx, chunks, n_past, 0, 1, true, &n_past);
    if (rc != 0) { llama_free(lctx); mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return ""; }

    // 개선된 샘플링: 더 많은 토큰 생성, 디버그 출력 추가
    std::string result;
    auto* vocab = llama_model_get_vocab(model);
    llama_token eos_token = llama_vocab_eos(vocab);
    
    std::cout << "Starting text generation..." << std::endl;
    
    for (int i = 0; i < 256; ++i) { // 더 많은 토큰 생성
        float* logits = llama_get_logits(lctx);
        if (!logits) {
            std::cout << "No logits available at step " << i << std::endl;
            break;
        }
        
        int n_vocab = llama_vocab_n_tokens(vocab);
        int max_id = 0;
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; ++j) {
            if (logits[j] > max_logit) { max_logit = logits[j]; max_id = j; }
        }
        
        // EOS 토큰이나 특정 종료 토큰 체크
        if (max_id == eos_token) {
            std::cout << "EOS token encountered at step " << i << std::endl;
            break;
        }
        
        // <|im_end|> 토큰 체크 (Qwen 모델의 경우)
        char buf[64] = {0};
        int n_chars = llama_token_to_piece(vocab, max_id, buf, sizeof(buf) - 1, 0, false);
        if (n_chars > 0) {
            std::string token_str(buf, n_chars);
            result += token_str;
            
            // 디버그: 처음 몇 개 토큰 출력
            if (i < 10) {
                std::cout << "Token " << i << ": '" << token_str << "' (id: " << max_id << ")" << std::endl;
            }
            
            // <|im_end|> 또는 <end_of_utterance> 토큰으로 종료
            if (token_str.find("<|im_end|>") != std::string::npos || 
                token_str.find("<end_of_utterance>") != std::string::npos) {
                std::cout << "Chat end token encountered at step " << i << std::endl;
                // 종료 토큰 제거
                size_t pos1 = result.find("<|im_end|>");
                size_t pos2 = result.find("<end_of_utterance>");
                if (pos1 != std::string::npos) {
                    result = result.substr(0, pos1);
                } else if (pos2 != std::string::npos) {
                    result = result.substr(0, pos2);
                }
                break;
            }
        }
        
        llama_token next_token = max_id;
        llama_batch next_batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(lctx, next_batch) != 0) {
            std::cout << "Decode failed at step " << i << std::endl;
            break;
        }
    }
    
    std::cout << "Generated " << result.length() << " characters" << std::endl;
    llama_free(lctx);
    mtmd_bitmap_free(bitmap);
    mtmd_input_chunks_free(chunks);
    mtmd_free(mmctx);
    llama_model_free(model);
    return result;
}

std::string LlmeService::queryImageBase64(const std::string& model_id, const std::string& text, const std::string& image_base64) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = running_models_.find(model_id);
    if (it == running_models_.end() || it->second.type != "multimodal") return "";
    const auto& m = it->second;

    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(m.model.c_str(), model_params);
    if (!model) return "";
    mtmd_context_params ctx_params = mtmd_context_params_default();
    mtmd_context* mmctx = mtmd_init_from_file(m.mmproj.c_str(), model, ctx_params);
    if (!mmctx) { llama_model_free(model); return ""; }

    // base64 decode (현재는 더미 처리)
    // TODO: 실제 base64 디코딩 구현 필요
    mtmd_bitmap* bitmap = nullptr;
    if (!bitmap) { mtmd_free(mmctx); llama_model_free(model); return "base64 decoding not implemented"; }

    // 텍스트에 이미지 마커 추가
    std::string prompt_with_marker = std::string("<__media__>\n") + text;
    mtmd_input_text input_text = { prompt_with_marker.c_str(), true, false };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[1] = { bitmap };
    if (mtmd_tokenize(mmctx, chunks, &input_text, bitmaps, 1) != 0) {
        mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return "";
    }

    llama_context_params lctx_params = llama_context_default_params();
    lctx_params.n_ctx = 4096;
    llama_context* lctx = llama_init_from_model(model, lctx_params);
    if (!lctx) { mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return ""; }

    llama_pos n_past = 0;
    int32_t rc = mtmd_helper_eval_chunks(mmctx, lctx, chunks, n_past, 0, 1, false, &n_past);
    if (rc != 0) { llama_free(lctx); mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return ""; }

    // 샘플링: greedy
    std::string result;
    auto* vocab = llama_model_get_vocab(model);
    for (int i = 0; i < 64; ++i) {
        float* logits = llama_get_logits(lctx);
        if (!logits) break;
        
        int n_vocab = llama_vocab_n_tokens(vocab);
        int max_id = 0;
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; ++j) {
            if (logits[j] > max_logit) { max_logit = logits[j]; max_id = j; }
        }
        
        llama_token eos_token = llama_vocab_eos(vocab);
        if (max_id == eos_token) break;
        
        char buf[32] = {0};
        int n_chars = llama_token_to_piece(vocab, max_id, buf, sizeof(buf) - 1, 0, false);
        if (n_chars > 0) {
            result += std::string(buf, n_chars);
        }
        
        llama_token next_token = max_id;
        llama_batch next_batch = llama_batch_get_one(&next_token, 1);
        llama_decode(lctx, next_batch);
    }
    llama_free(lctx);
    mtmd_bitmap_free(bitmap);
    mtmd_input_chunks_free(chunks);
    mtmd_free(mmctx);
    llama_model_free(model);
    return result;
}
