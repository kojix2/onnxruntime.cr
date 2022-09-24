Usage

```
docker build -t c2ffi .
wget https://raw.githubusercontent.com/microsoft/onnxruntime/main/include/onnxruntime/core/session/onnxruntime_c_api.h
docker run --rm -v $(pwd):/data c2ffi onnxruntime_c_api.h > capi.json
```