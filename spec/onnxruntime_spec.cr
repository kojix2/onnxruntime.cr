require "./spec_helper"

describe OnnxRuntime do
  # TODO: Write tests

  it "works" do
    a = OnnxRuntime::LibOnnxRuntime.OrtGetApiBase
    p a
    false.should eq("hoge")
  end
end
