samples/dbus-mtmd 에 dbus 서비스 하나를 만들어.
아래는 요구사항이야.

1. libggml, libllama 와 libmtmd 를 이용하여 만들어.
2. libggml., libllama 와 libmtmd 는  /usr/local/lib/ 하위에 so 파일이 존재해.
```
/usr/local/lib/libggml-base.so
/usr/local/lib/libllama.so
/usr/local/lib/libmtmd.so
```
3. libggml, libllama와 libmtmd 는 /usr/local/include/ 하위에 .h 가 존재해
```
/usr/local/include/ggml.h
/usr/local/include/ggml-cpu.h
/usr/local/include/ggml-alloc.h
/usr/local/include/ggml-backend.h
/usr/local/include/ggml-blas.h
/usr/local/include/ggml-cann.h
/usr/local/include/ggml-cpp.h
/usr/local/include/ggml-cuda.h
/usr/local/include/ggml-kompute.h
/usr/local/include/ggml-opt.h
/usr/local/include/ggml-metal.h
/usr/local/include/ggml-rpc.h
/usr/local/include/ggml-sycl.h
/usr/local/include/ggml-vulkan.h
/usr/local/include/gguf.h

/usr/local/include/llama.h
/usr/local/include/llama-cpp.h

/usr/local/include/mtmd.h
/usr/local/include/mtmd-helper.h
```
4. sample/dbus-mtmd 하위의 CMakeLists.txt 에서는 위 library 를 찾아서 header 참조와 library link 를 설정해야해. () libmtmd 는 별도 cmake 가 없으니  libmtmd.so 를 링크해야 할 것 같네.)
```
/usr/local/lib/cmake/ggml/ggml-config.cmake
/usr/local/lib/cmake/ggml/ggml-version.cmake

/usr/local/lib/cmake/llama/llama-config.cmake
/usr/local/lib/cmake/llama/llama-version.cmake
```
5. 서비스는 다음과 같은 dbus 설정 을 가지도록 해.
service name: ecoai.llme
object path: /ecoai/llme

6. 서비스에서 제공하는 inteface 는 총 3가지야.
6.1.  ecoai.llme.manage
이것은 ecoai.llme 서비스를 제어하는 인터페이스야 (서비스 상태)

methods of ecoai.llme.manage

 .getStatus  (서비스 초기화 중, 로드된 모델 없음, 모델 로드 중(모델 정보 포함), 모델 로드 완료(모델 정보 포함), 모델 실행 중(모델 정보 포함), 모델 언로드 중(모델 정보 포함), 서비스 종료 중 )

6.2.  ecoai.llme.model
이것은 모델을 로드/실행하는  인터페이스야 ( 모델 정보, 모델 로드/언로드, 모델 실행 등)

methods of ecoai.llme.model interface

 .getModels   (사용 가능한 모델 리스트를 보여 줌)
     이 정보는 특정 디렉토리에 존재하는 json 파일을 기준으로 정보를 리턴함
     아래는 json sample (/tmp/models.json)
     ```
    {
        {
            "type":"embedding",
            "name":"embed-model-v1",
            "model": "/tmp/emb/embed.gguf" 
        },
        {
            "type":"llm",
            "name":"llama3.2-1b",
            "model":"/tmp/llama3.2-1b/model.gguf"
        },
        {
            "type":"multimodal",
            "name":"gemma3-4b",
            "model":"/tmp/gemma3-4b-it/gemma-3-4b-it-Q4_K_M.gguf",
            "mmproj":"/tmp/gemma3-4b-it/mmproj-model-f16.gguf"
        }
    }
     ```
 .cancel   (실행 중인 모델 정리 또는 실행 취소하고 리턴 true,  실행 중인 모델이 없으면 그냥 리턴 true)
 .load  (model name 을 받아서 실행,  multimodal 인 경우 mtmd 를 사용하여 model 와 mmproj 를 로드, 로드된 모델은 유익한 model-id 를 생성하여 리턴할 때 모델 정보와 함께 리턴해)
 .unload  (로드된 모델 언로드)
 .getRunningModel  (로드된 모델 리스트 정보를 리턴해, 각 모델 정보에는 model-id 가 있어.)
 .embed (실행 시 model-id 와 text 를 넘겨줘야 해. 이것은 실행된 모델이 embed 모델인 경우만 사용가능해,  model 을 실행 후 1차원 f16 array를 리턴하면 돼)
 .query 
   (실행 시 model-id와 text 를 넘겨줘야 해. 이것은 실행된 모델이 llm 또는 multimodal 모델인 경우만 사용 가능해. model 을 실행 후 Generated Text 를 리턴해야 해.)
.queryImage
  (실행 시 model-id와 text, image file path 를 넘겨 줘야 해. 이것은 실행된 모델이 multimodal 모델인 경우만 사용 가능해. model 실행 후 Generated Text 를 리턴해야 해.)
.queryImageBase64
 (실행 시 model-id와 text, image file data (base64 encoded data) 를 넘겨 줘야 해. 이것은 실행된 모델이 multimodal 모델인 경우만 사용 가능해. model 실행 후 Generated Text 를 리턴해야 해. )

 7. dbus service 는 sdbus-c++ library 를 사용해서 구현해.
 