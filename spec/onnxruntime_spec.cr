require "./spec_helper"

describe OnnxRuntime do
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
  it "create" do
    session = OnnxRuntime::InferenceSession.new("spec/fixtures/mnist.onnx")
    session.should be_a(OnnxRuntime::InferenceSession)
    session.inputs.should be_a(Array(NamedTuple(name: String, type: OnnxRuntime::LibOnnxRuntime::TensorElementDataType, shape: Array(Int64))))
    session.outputs.should be_a(Array(NamedTuple(name: String, type: OnnxRuntime::LibOnnxRuntime::TensorElementDataType, shape: Array(Int64))))
  end
end

describe OnnxRuntime::Model do
  it "create" do
    model = OnnxRuntime::Model.new("spec/fixtures/mnist.onnx")
    model.should be_a(OnnxRuntime::Model)
    model.inputs.should be_a(Array(NamedTuple(name: String, type: OnnxRuntime::LibOnnxRuntime::TensorElementDataType, shape: Array(Int64))))
    model.outputs.should be_a(Array(NamedTuple(name: String, type: OnnxRuntime::LibOnnxRuntime::TensorElementDataType, shape: Array(Int64))))
  end

  it "predict" do
    model = OnnxRuntime::Model.new("spec/fixtures/mnist.onnx")

    # MNISTモデルの入力は28x28のグレースケール画像
    # ダミーの入力データを作成（すべて0の画像）
    input_data = Array(Float32).new(28 * 28, 0.0_f32)

    # 予測を実行
    result = model.predict({"Input3" => input_data})

    # 結果は10クラス（0-9の数字）の確率分布
    result.should be_a(Hash(String, Array(Float32) | Array(Int32) | Array(Int64) | Array(Bool) | Array(UInt8) | Array(Int8) | Array(UInt16) | Array(Int16) | Array(UInt32) | Array(UInt64) | OnnxRuntime::SparseTensor))
    result.keys.should contain("Plus214_Output_0")

    # 出力は10個の値（各数字の確率）を持つ配列
    output = result["Plus214_Output_0"]
    output.is_a?(Array(Float32)).should be_true
    if output.is_a?(Array(Float32))
      output.size.should eq(10)
    end
  end
end
