#include <dbus/dbus.h>
#include <string>
#include <iostream>
#include <stdexcept>

class LlamaDBusClient {
private:
    DBusConnection* conn;

public:
    LlamaDBusClient() : conn(nullptr) {
        DBusError err;
        dbus_error_init(&err);
        
        conn = dbus_bus_get(DBUS_BUS_SESSION, &err);
        if (dbus_error_is_set(&err)) {
            throw std::runtime_error("Failed to get DBus connection: " + std::string(err.message));
        }
    }

    ~LlamaDBusClient() {
        if (conn) {
            dbus_connection_unref(conn);
        }
    }

    std::string generate_text(const std::string& prompt, int n_predict = 128) {
        DBusMessage* msg = dbus_message_new_method_call(
            "org.llama.cpp",           // destination
            "/org/llama/cpp",          // path
            "org.llama.cpp",           // interface
            "GenerateText"             // method
        );

        if (!msg) {
            throw std::runtime_error("Failed to create DBus message");
        }

        if (!dbus_message_append_args(msg,
            DBUS_TYPE_STRING, &prompt,
            DBUS_TYPE_INT32, &n_predict,
            DBUS_TYPE_INVALID)) {
            dbus_message_unref(msg);
            throw std::runtime_error("Failed to append arguments");
        }

        DBusError err;
        dbus_error_init(&err);
        
        DBusMessage* reply = dbus_connection_send_with_reply_and_block(conn, msg, -1, &err);
        dbus_message_unref(msg);

        if (dbus_error_is_set(&err)) {
            std::string error = err.message;
            dbus_error_free(&err);
            throw std::runtime_error("Failed to send message: " + error);
        }

        if (!reply) {
            throw std::runtime_error("No reply received");
        }

        const char* result;
        if (!dbus_message_get_args(reply, &err,
            DBUS_TYPE_STRING, &result,
            DBUS_TYPE_INVALID)) {
            std::string error = err.message;
            dbus_error_free(&err);
            dbus_message_unref(reply);
            throw std::runtime_error("Failed to get reply arguments: " + error);
        }

        std::string text = result;
        dbus_message_unref(reply);
        return text;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <prompt>" << std::endl;
        return 1;
    }

    try {
        LlamaDBusClient client;
        std::string result = client.generate_text(argv[1]);
        std::cout << "Generated text: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 