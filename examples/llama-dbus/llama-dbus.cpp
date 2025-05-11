#include "common.h"
#include "llama.h"
#include "chat.h"
#include <dbus/dbus.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

class LlamaDBusService {
private:
    DBusConnection* conn;
    llama_model* model;
    llama_context* ctx;
    common_sampler* smpl;
    common_params params;
    std::vector<llama_token> session_tokens;
    std::vector<common_chat_msg> chat_msgs;
    std::unique_ptr<common_chat_templates> chat_templates;

public:
    LlamaDBusService() : conn(nullptr), model(nullptr), ctx(nullptr), smpl(nullptr) {
        // Initialize DBus connection
        DBusError err;
        dbus_error_init(&err);
        
        conn = dbus_bus_get(DBUS_BUS_SESSION, &err);
        if (dbus_error_is_set(&err)) {
            throw std::runtime_error("Failed to get DBus connection: " + std::string(err.message));
        }

        // Request name on the bus
        int ret = dbus_bus_request_name(conn, "org.llama.cpp", DBUS_NAME_FLAG_REPLACE_EXISTING, &err);
        if (dbus_error_is_set(&err)) {
            throw std::runtime_error("Failed to request name: " + std::string(err.message));
        }
        if (ret != DBUS_REQUEST_NAME_REPLY_PRIMARY_OWNER) {
            throw std::runtime_error("Failed to get primary owner");
        }
    }

    ~LlamaDBusService() {
        if (smpl) {
            common_sampler_free(smpl);
        }
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_free_model(model);
        }
        if (conn) {
            dbus_connection_unref(conn);
        }
    }

    void load_model(const std::string& model_path) {
        params.model = model_path;
        common_init_result llama_init = common_init_from_params(params);
        model = llama_init.model.get();
        ctx = llama_init.context.get();
        
        if (!model || !ctx) {
            throw std::runtime_error("Failed to load model");
        }

        chat_templates = common_chat_templates_init(model, params.chat_template);
        smpl = common_sampler_init(model, params.sampling);
    }

    std::string generate_text(const std::string& prompt, int n_predict = 128) {
        if (!model || !ctx || !smpl) {
            throw std::runtime_error("Model not loaded");
        }

        std::vector<llama_token> embd_inp = common_tokenize(ctx, prompt, true, true);
        
        if (embd_inp.empty()) {
            throw std::runtime_error("Empty prompt");
        }

        std::string result;
        int n_past = 0;
        int n_remain = n_predict;

        while (n_remain > 0) {
            if (n_past + (int)embd_inp.size() > llama_n_ctx(ctx)) {
                break;
            }

            if (llama_decode(ctx, llama_batch_get_one(embd_inp.data(), embd_inp.size()))) {
                throw std::runtime_error("Failed to decode");
            }

            n_past += embd_inp.size();
            embd_inp.clear();

            const llama_token id = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, id, true);

            if (llama_vocab_is_eog(llama_model_get_vocab(model), id)) {
                break;
            }

            result += common_token_to_piece(ctx, id);
            embd_inp.push_back(id);
            n_remain--;
        }

        return result;
    }

    void run() {
        while (true) {
            dbus_connection_read_write(conn, 0);
            DBusMessage* msg = dbus_connection_pop_message(conn);
            
            if (!msg) {
                continue;
            }

            if (dbus_message_is_method_call(msg, "org.llama.cpp", "GenerateText")) {
                DBusError err;
                dbus_error_init(&err);
                
                const char* prompt;
                int n_predict;
                
                if (!dbus_message_get_args(msg, &err, DBUS_TYPE_STRING, &prompt,
                                         DBUS_TYPE_INT32, &n_predict,
                                         DBUS_TYPE_INVALID)) {
                    DBusMessage* reply = dbus_message_new_error(msg, err.name, err.message);
                    dbus_connection_send(conn, reply, nullptr);
                    dbus_message_unref(reply);
                    continue;
                }

                try {
                    std::string result = generate_text(prompt, n_predict);
                    DBusMessage* reply = dbus_message_new_method_return(msg);
                    dbus_message_append_args(reply, DBUS_TYPE_STRING, &result,
                                          DBUS_TYPE_INVALID);
                    dbus_connection_send(conn, reply, nullptr);
                    dbus_message_unref(reply);
                } catch (const std::exception& e) {
                    DBusMessage* reply = dbus_message_new_error(msg,
                        "org.llama.cpp.Error", e.what());
                    dbus_connection_send(conn, reply, nullptr);
                    dbus_message_unref(reply);
                }
            }
            
            dbus_message_unref(msg);
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    try {
        LlamaDBusService service;
        service.load_model(argv[1]);
        service.run();
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
} 