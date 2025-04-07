require "spec"
require "../src/onnxruntime"

# Ensure environment is released after all tests
Spec.after_suite do
  OnnxRuntime::OrtEnvironment.instance.release
end
