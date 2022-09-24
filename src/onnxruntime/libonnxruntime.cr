module OnnxRuntime
  {% if env("ONNXRUNTIMEDIR") %}
    @[Link(ldflags: "-L `echo $ONNXRUNTIMEDIR/lib` -lonnxruntime -Wl,-rpath,`echo $ONNXRUNTIMEDIR/lib`")]
  {% else %}
    @[Link("onnxruntime")]
  {% end %}
  lib LibOnnxRuntime
    enum TensorElementDataType
      Undefined
      Float
      UInt8
      Int8
      UInt16
      Int16
      Int32
      Int64
      String
      Bool
      Float16
      Double
      UInt32
      UInt64
      Complex64
      Complex128
      BFloat16
    end

    enum OnnxType
      Unknown
      Tensor
      Sequence
      Map
      Opaque
      SparseTensor
    end

    struct ApiBase
      get_api : Void*
      get_version_string : (-> LibC::Char*)
    end

    fun OrtGetApiBase : ApiBase*
  end
end
