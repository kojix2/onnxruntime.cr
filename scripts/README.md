Usage

```
docker build -t c2ffi .
wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.3/include/onnxruntime/core/session/onnxruntime_c_api.h
docker run --rm -v $(pwd):/data c2ffi onnxruntime_c_api.h > capi.json
```
