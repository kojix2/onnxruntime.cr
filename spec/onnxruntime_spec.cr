require "./spec_helper"

describe OnnxRuntime do
  # TODO: Write tests

  it "has a version number" do
    (OnnxRuntime::VERSION).should_not be_nil
  end

  it "version" do
    base = OnnxRuntime::LibOnnxRuntime.OrtGetApiBase
    v = String.new(base.value.get_version_string.call)
    v.should be_a(String)
  end
end

describe OnnxRuntime::InferenceSession do
#  it "create" do
#    session = OnnxRuntime::InferenceSession.new("spec/fixtures/mnist.onnx")
#    session.should be_a(InferenceSession)
#  end
end
