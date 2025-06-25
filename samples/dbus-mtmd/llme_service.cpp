#include "llme_service.hpp"
#include <fstream>
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
    for (int i = 0; i < 64; ++i) {
        float* logits = llama_get_logits(ctx);
        int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        int max_id = 0;
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; ++j) {
            if (logits[j] > max_logit) { max_logit = logits[j]; max_id = j; }
        }
        if (max_id == llama_vocab_eos(llama_model_get_vocab(model))) break;
        char buf[16] = {0};
        llama_token_to_piece(llama_model_get_vocab(model), max_id, buf, sizeof(buf), 0, false);
        result += buf;
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
    llama_model* model = llama_model_load_from_file(m.model.c_str(), model_params);
    if (!model) return "";
    mtmd_context_params ctx_params = mtmd_context_params_default();
    mtmd_context* mmctx = mtmd_init_from_file(m.mmproj.c_str(), model, ctx_params);
    if (!mmctx) { llama_model_free(model); return ""; }

    mtmd_bitmap* bitmap = mtmd_helper_bitmap_init_from_file(mmctx, image_path.c_str());
    if (!bitmap) { mtmd_free(mmctx); llama_model_free(model); return ""; }

    mtmd_input_text input_text = { text.c_str(), true, false };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[1] = { bitmap };
    if (mtmd_tokenize(mmctx, chunks, &input_text, bitmaps, 1) != 0) {
        mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return "";
    }

    llama_context_params lctx_params = llama_context_default_params();
    llama_context* lctx = llama_init_from_model(model, lctx_params);
    if (!lctx) { mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return ""; }

    llama_pos n_past = 0;
    int32_t rc = mtmd_helper_eval_chunks(mmctx, lctx, chunks, n_past, 0, 1, false, &n_past);
    if (rc != 0) { llama_free(lctx); mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return ""; }

    // 샘플링: greedy
    std::string result;
    for (int i = 0; i < 64; ++i) {
        float* logits = llama_get_logits(lctx);
        int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        int max_id = 0;
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; ++j) {
            if (logits[j] > max_logit) { max_logit = logits[j]; max_id = j; }
        }
        if (max_id == llama_vocab_eos(llama_model_get_vocab(model))) break;
        char buf[16] = {0};
        llama_token_to_piece(llama_model_get_vocab(model), max_id, buf, sizeof(buf), 0, false);
        result += buf;
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

    // base64 decode
    std::string decoded;
    // (base64 decode 함수는 별도 구현 필요, 예시에서는 생략)
    // decoded = base64_decode(image_base64);
    // unsigned char* buf = reinterpret_cast<unsigned char*>(decoded.data());
    // size_t len = decoded.size();
    // mtmd_bitmap* bitmap = mtmd_helper_bitmap_init_from_buf(mmctx, buf, len);
    // 아래는 더미 처리
    mtmd_bitmap* bitmap = nullptr;
    if (!bitmap) { mtmd_free(mmctx); llama_model_free(model); return ""; }

    mtmd_input_text input_text = { text.c_str(), true, false };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps[1] = { bitmap };
    if (mtmd_tokenize(mmctx, chunks, &input_text, bitmaps, 1) != 0) {
        mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return "";
    }

    llama_context_params lctx_params = llama_context_default_params();
    llama_context* lctx = llama_init_from_model(model, lctx_params);
    if (!lctx) { mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return ""; }

    llama_pos n_past = 0;
    int32_t rc = mtmd_helper_eval_chunks(mmctx, lctx, chunks, n_past, 0, 1, false, &n_past);
    if (rc != 0) { llama_free(lctx); mtmd_bitmap_free(bitmap); mtmd_input_chunks_free(chunks); mtmd_free(mmctx); llama_model_free(model); return ""; }

    // 샘플링: greedy
    std::string result;
    for (int i = 0; i < 64; ++i) {
        float* logits = llama_get_logits(lctx);
        int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        int max_id = 0;
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; ++j) {
            if (logits[j] > max_logit) { max_logit = logits[j]; max_id = j; }
        }
        if (max_id == llama_vocab_eos(llama_model_get_vocab(model))) break;
        char buf[16] = {0};
        llama_token_to_piece(llama_model_get_vocab(model), max_id, buf, sizeof(buf), 0, false);
        result += buf;
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
