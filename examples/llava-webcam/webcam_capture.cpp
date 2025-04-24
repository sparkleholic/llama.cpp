#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <filesystem>

int main() {
    // Open the default camera (usually /dev/video0 on Linux)
    cv::VideoCapture cap(0);
    
    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Create output directory if it doesn't exist
    std::filesystem::path outputDir = "captured_frames";
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directory(outputDir);
    }

    int frameCount = 0;
    auto lastCaptureTime = std::chrono::steady_clock::now();

    std::cout << "Starting frame capture. Press 'q' to quit." << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed" << std::endl;
            break;
        }

        // Show live preview
        cv::imshow("Webcam Preview", frame);

        // Get current time
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>
            (currentTime - lastCaptureTime).count();

        // If one second has passed, save the frame
        if (elapsedTime >= 1000) {
            std::string filename = "captured_frames/frame_" + 
                                 std::to_string(frameCount++) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "Captured: " << filename << std::endl;
            lastCaptureTime = currentTime;
        }

        // Break the loop if 'q' is pressed
        char c = (char)cv::waitKey(1);
        if (c == 'q' || c == 'Q')
            break;
    }

    // Release the camera and destroy windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
} 