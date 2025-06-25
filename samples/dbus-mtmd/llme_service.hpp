#pragma once
#include <sdbus-c++/sdbus-c++.h>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <optional>
#include <nlohmann/json.hpp>

struct ModelInfo {
    std::string type;
    std::string name;
    std::string model;
    std::string mmproj;
    std::string model_id;
};

class LlmeService {
public:
    LlmeService();

    // ecoai.llme.manage
    std::string getStatus();

    // ecoai.llme.model
    std::vector<nlohmann::json> getModels();
    bool cancel();
    nlohmann::json load(const std::string& name);
    bool unload(const std::string& model_id);
    std::vector<nlohmann::json> getRunningModel();
    std::vector<float> embed(const std::string& model_id, const std::string& text);
    std::string query(const std::string& model_id, const std::string& text);
    std::string queryImage(const std::string& model_id, const std::string& text, const std::string& image_path);
    std::string queryImageBase64(const std::string& model_id, const std::string& text, const std::string& image_base64);

private:
    std::mutex mtx_;
    std::string status_;
    std::vector<ModelInfo> available_models_;
    std::map<std::string, ModelInfo> running_models_;
    std::optional<ModelInfo> current_model_;
    void loadModelsFromJson();
    std::string generateModelId(const std::string& name);
};
