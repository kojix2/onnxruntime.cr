module OnnxRuntime
  {% if env("ONNXRUNTIMEDIR") %}
    @[Link(ldflags: "-L `echo $ONNXRUNTIMEDIR/lib` -lGR -Wl,-rpath,`echo $ONNXRUNTIMEDIR/lib`")]
  {% else %}
    @[Link("onnxruntime")]
  {% end %}
  lib LibOnnxRuntime
    struct ApiBase
      get_api : Void*
      get_version_string : (Void* -> LibC::Char*)
    end

    fun OrtGetApiBase() : ApiBase*
  end
end