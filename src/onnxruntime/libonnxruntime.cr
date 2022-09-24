module Onnxruntime
  {% if env("ONNXRUNTIMEDIR") %}
    @[Link(ldflags: "-L `echo $ONNXRUNTIMEDIR/lib` -lGR -Wl,-rpath,`echo $ONNXRUNTIMEDIR/lib`")]
  {% else %}
    @[Link("onnxruntime")]
  {% end %}
  lib LibOnnxRuntime
  end
end