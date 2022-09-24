require "./spec_helper"

describe OnnxRuntime do
  # TODO: Write tests

  it "works" do
    base = OnnxRuntime::LibOnnxRuntime.OrtGetApiBase
    p String.new(base.value.get_version_string.call)
    false.should eq("version")
  end
end
