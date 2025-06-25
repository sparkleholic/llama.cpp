#include "llme_service.hpp"
#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::cout << "=== LlmeService queryImage Test ===" << std::endl;
    
    // LlmeService 인스턴스 생성
    LlmeService service;
    
    // 상태 확인
    std::cout << "Initial status: " << service.getStatus() << std::endl;
    
    // 사용 가능한 모델 조회
    auto models = service.getModels();
    std::cout << "\nAvailable models:" << std::endl;
    for (const auto& model : models) {
        std::cout << "  - " << model["name"].get<std::string>() 
                  << " (type: " << model["type"].get<std::string>() << ")" << std::endl;
    }
    
    // 멀티모달 모델 로드 (SmolVLM 사용해보기)
    std::string model_name = "smolvlm-500m";
    std::cout << "\nLoading model: " << model_name << std::endl;
    auto load_result = service.load(model_name);
    
    if (load_result.empty()) {
        std::cout << "Failed to load model: " << model_name << std::endl;
        return 1;
    }
    
    std::string model_id = load_result["model_id"].get<std::string>();
    std::cout << "Model loaded successfully with ID: " << model_id << std::endl;
    
    // 이미지 경로와 텍스트 설정
    std::string image_path = "/hdd/Project/llama.cpp/samples/dbus-mtmd/cat.jpg";
    std::string query_text = "이 이미지를 자세히 설명해주세요.";
    
    std::cout << "\nTesting queryImage..." << std::endl;
    std::cout << "Image path: " << image_path << std::endl;
    std::cout << "Query text: " << query_text << std::endl;
    
    // queryImage 함수 테스트
    std::string result = service.queryImage(model_id, query_text, image_path);
    
    std::cout << "\n=== Result ===" << std::endl;
    if (result.empty()) {
        std::cout << "No result returned (empty string)" << std::endl;
    } else {
        std::cout << "Response: " << result << std::endl;
    }
    
    // 모델 언로드
    std::cout << "\nUnloading model..." << std::endl;
    bool unload_success = service.unload(model_id);
    std::cout << "Unload result: " << (unload_success ? "Success" : "Failed") << std::endl;
    
    // 최종 상태 확인
    std::cout << "Final status: " << service.getStatus() << std::endl;
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
