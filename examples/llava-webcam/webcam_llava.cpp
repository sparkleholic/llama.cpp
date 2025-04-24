#include <string>
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "sampling.h"
#include "chat.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

static volatile bool g_is_running = true;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        g_is_running = false;
    }
}
#endif

class LlavaWebcam {
private:
    mtmd_context_ptr ctx_vision;
    llama_model* model;
    llama_context* lctx;
    const llama_vocab* vocab;
    llama_batch batch;
    common_chat_templates_ptr tmpls;
    int n_threads;
    llama_pos n_past;
    cv::VideoCapture cap;
    std::string last_frame_path;
    std::chrono::steady_clock::time_point last_capture_time;
    const int CAPTURE_INTERVAL_MS = 1000; // 1 second interval

    void init_vision_context(const char* mmproj_path, const char* model_path) {
        common_params params;
        params.model.path = model_path;
        params.mmproj.path = mmproj_path;
        params.cpuparams.n_threads = 4;
        params.n_batch = 512;
        params.chat_template = "deepseek";

        auto llama_init = common_init_from_params(params);
        model = llama_init.model.get();
        lctx = llama_init.context.get();
        
        if (!model) {
            throw std::runtime_error("Failed to load language model");
        }
        if (!lctx) {
            throw std::runtime_error("Failed to create llama context");
        }
        std::cout << "[DEBUG] Llama model and context initialized successfully." << std::endl;

        vocab = llama_model_get_vocab(model);
        
        if (!vocab) {
             throw std::runtime_error("Failed to get vocabulary from model");
        }
        std::cout << "[DEBUG] Vocabulary obtained successfully." << std::endl;

        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(params.n_batch, 0, 1);

        ctx_vision.reset(mtmd_init_from_file(mmproj_path, model, mtmd_context_params{
            true,  // use_gpu
            true,  // timings
            n_threads,
            GGML_LOG_LEVEL_INFO,
        }));

        if (!ctx_vision.get()) {
            throw std::runtime_error("Failed to initialize vision context");
        }

        tmpls = common_chat_templates_init(model, params.chat_template);
    }

    bool process_frame(const cv::Mat& frame) {
        std::cout << "[DEBUG] Entering process_frame" << std::endl;
        
        // Encode frame to JPEG buffer in memory
        std::vector<unsigned char> buf;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
        std::cout << "[DEBUG] Encoding frame to JPEG buffer..." << std::endl;
        if (!cv::imencode(".jpg", frame, buf, params)) {
            std::cerr << "[ERROR] Failed to encode frame to JPEG buffer" << std::endl;
            return false;
        }
        std::cout << "[DEBUG] Frame encoded to buffer (size: " << buf.size() << ")." << std::endl;
        
        // Create bitmap from memory buffer
        mtmd_bitmap bitmap;
        std::cout << "[DEBUG] Initializing bitmap from buffer..." << std::endl;
        if (mtmd_helper_bitmap_init_from_buf(buf.data(), buf.size(), bitmap) != 0) {
            std::cerr << "[ERROR] Failed to create bitmap from buffer" << std::endl;
            return false;
        }
        std::cout << "[DEBUG] Bitmap initialized from buffer." << std::endl;

        // --- DEBUG: Test text-only tokenization ---
        mtmd_input_text text_only_input;
        text_only_input.text = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Describe what you see.\nASSISTANT:"; // Removed <__image__>
        text_only_input.add_special = true;
        text_only_input.parse_special = true;
        std::cout << "[DEBUG] Text-only input prepared." << std::endl;

        // Tokenize input (text only)
        std::vector<mtmd_input_chunk> chunks;
        std::cout << "[DEBUG] Tokenizing text-only input..." << std::endl;
        // Pass an empty vector for bitmaps
        if (mtmd_tokenize(ctx_vision.get(), chunks, text_only_input, {}) != 0) { 
            std::cerr << "[ERROR] Failed to tokenize text-only input" << std::endl;
            // Continue for now, maybe the rest works? Or return false?
            // Let's try continuing to see if eval works
            // return false; 
        }
        std::cout << "[DEBUG] Text-only input tokenized. Number of chunks: " << chunks.size() << std::endl;
        // --- END DEBUG ---

        // Process chunks (using the text-only chunks from the debug section)
        n_past = 0;
        std::cout << "[DEBUG] Evaluating input chunks (n_past=0)..." << std::endl;
        if (mtmd_helper_eval(ctx_vision.get(), lctx, chunks, 0, 0, 512) != 0) {
            std::cerr << "[ERROR] Failed to evaluate input" << std::endl;
            return false;
        }
        std::cout << "[DEBUG] Input chunks evaluated." << std::endl;

        // Sample and generate response
        common_params_sampling sparams;
        sparams.temp = 0.7f;
        sparams.top_k = 40;
        sparams.top_p = 0.9f;
        sparams.penalty_repeat = 1.1f;
        std::cout << "[DEBUG] Sampling parameters set." << std::endl;

        // Initialize sampler AFTER setting all parameters
        std::cout << "[DEBUG] Initializing sampler..." << std::endl;
        auto* sampler = common_sampler_init(model, sparams);
        if (!sampler) {
            std::cerr << "[ERROR] Failed to initialize sampler" << std::endl;
            return false;
        }
        std::cout << "[DEBUG] Sampler initialized." << std::endl;

        std::cout << "\nResponse: ";
        std::cout << "[DEBUG] Generating response..." << std::endl;
        generate_response(sampler, 100);  // Generate up to 100 tokens
        std::cout << "[DEBUG] Response generated." << std::endl;
        common_sampler_free(sampler);
        std::cout << "[DEBUG] Sampler freed." << std::endl;

        std::cout << "[DEBUG] Exiting process_frame successfully." << std::endl;
        return true;
    }

    void generate_response(common_sampler* sampler, int n_predict) {
        std::vector<llama_token> generated_tokens;
        for (int i = 0; i < n_predict && g_is_running; i++) {
            llama_token token_id = common_sampler_sample(sampler, lctx, -1);
            generated_tokens.push_back(token_id);
            common_sampler_accept(sampler, token_id, true);

            if (llama_vocab_is_eog(vocab, token_id)) {
                break;
            }

            std::string piece = common_token_to_piece(lctx, token_id);
            std::cout << piece << std::flush;

            // Decode the token
            common_batch_clear(batch);
            common_batch_add(batch, token_id, n_past++, {0}, true);
            if (llama_decode(lctx, batch)) {
                std::cerr << "Failed to decode token" << std::endl;
                break;
            }
        }
        std::cout << std::endl;
    }

public:
    LlavaWebcam(const char* mmproj_path, const char* model_path) {
        init_vision_context(mmproj_path, model_path);
        cap.open(0);
        if (!cap.isOpened()) {
            throw std::runtime_error("Could not open camera");
        }
        last_capture_time = std::chrono::steady_clock::now();
    }

    ~LlavaWebcam() {
        cap.release();
        cv::destroyAllWindows();
        if (!last_frame_path.empty() && std::filesystem::exists(last_frame_path)) {
            std::filesystem::remove(last_frame_path);
        }
    }

    void run() {
        std::cout << "Starting webcam capture and analysis. Press Ctrl+C to exit." << std::endl;

        while (g_is_running) {
            cv::Mat frame;
            cap >> frame;

            if (frame.empty()) {
                std::cerr << "Error: Blank frame grabbed" << std::endl;
                break;
            }

            // Show live preview
            cv::imshow("Webcam Preview", frame);
            char c = (char)cv::waitKey(1);
            if (c == 'q' || c == 'Q') {
                break;
            }

            // Process frame every second
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>
                (current_time - last_capture_time).count();

            if (elapsed_time >= CAPTURE_INTERVAL_MS) {
                std::cout << "\nProcessing new frame..." << std::endl;
                if (!process_frame(frame)) {
                    std::cerr << "Failed to process frame" << std::endl;
                }
                last_capture_time = current_time;
            }
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <mmproj_path>" << std::endl;
        return 1;
    }

    // Set up signal handler
    #if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        signal(SIGINT, sigint_handler);
    #endif

    try {
        LlavaWebcam llava(argv[2], argv[1]);
        llava.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 