#include "llme_service.hpp"
#include <sdbus-c++/sdbus-c++.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <cstring>

using json = nlohmann::json;

int main() {
    auto connection = sdbus::createSessionBusConnection();
    connection->requestName("ecoai.llme");
    auto object = sdbus::createObject(*connection, "/ecoai/llme");
    LlmeService service;

    // ecoai.llme.manage
    object->registerMethod("getStatus")
        .onInterface("ecoai.llme.manage")
        .implementedAs([&service]() {
            return service.getStatus();
        });

    // ecoai.llme.model
    object->registerMethod("getModels")
        .onInterface("ecoai.llme.model")
        .implementedAs([&service]() {
            auto models = service.getModels();
            return json(models).dump();
        });
    object->registerMethod("cancel")
        .onInterface("ecoai.llme.model")
        .implementedAs([&service]() {
            return service.cancel();
        });
    object->registerMethod("load")
        .onInterface("ecoai.llme.model")
        .implementedAs([&service](const std::string& name) {
            auto result = service.load(name);
            return result.dump();
        });
    object->registerMethod("unload")
        .onInterface("ecoai.llme.model")
        .implementedAs([&service](const std::string& model_id) {
            return service.unload(model_id);
        });
    object->registerMethod("getRunningModel")
        .onInterface("ecoai.llme.model")
        .implementedAs([&service]() {
            auto running = service.getRunningModel();
            return json(running).dump();
        });
    object->registerMethod("embed")
        .onInterface("ecoai.llme.model")
        .implementedAs([&service](const std::string& model_id, const std::string& text) {
            auto arr = service.embed(model_id, text);
            std::vector<uint8_t> bytes(arr.size() * sizeof(float));
            std::memcpy(bytes.data(), arr.data(), bytes.size());
            return bytes;
        });
    object->registerMethod("query")
        .onInterface("ecoai.llme.model")
        .implementedAs([&service](const std::string& model_id, const std::string& text) {
            return service.query(model_id, text);
        });
    object->registerMethod("queryImage")
        .onInterface("ecoai.llme.model")
        .implementedAs([&service](const std::string& model_id, const std::string& text, const std::string& image_path) {
            return service.queryImage(model_id, text, image_path);
        });
    object->registerMethod("queryImageBase64")
        .onInterface("ecoai.llme.model")
        .implementedAs([&service](const std::string& model_id, const std::string& text, const std::string& image_base64) {
            return service.queryImageBase64(model_id, text, image_base64);
        });

    object->finishRegistration();
    connection->enterEventLoop();
    return 0;
}
