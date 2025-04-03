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

    # MNISTモデルの入力は[バッチサイズ, チャンネル数, 高さ, 幅]の4次元
    # ダミーの入力データを作成（すべて0の画像）
    input_data = Array(Float32).new(1 * 1 * 28 * 28, 0.0_f32)

    # 予測を実行（4次元の形状を指定）
    result = model.predict({"Input3" => input_data}, nil, shape: {"Input3" => [1_i64, 1_i64, 28_i64, 28_i64]})

    # 結果は10クラス（0-9の数字）の確率分布
    result.should be_a(Hash(String, Array(Float32) | Array(Int32) | Array(Int64) | Array(Bool) | Array(UInt8) | Array(Int8) | Array(UInt16) | Array(Int16) | Array(UInt32) | Array(UInt64) | OnnxRuntime::SparseTensorFloat32 | OnnxRuntime::SparseTensorInt32 | OnnxRuntime::SparseTensorInt64 | OnnxRuntime::SparseTensorFloat64))
    result.keys.should contain("Plus214_Output_0")

    # 出力は10個の値（各数字の確率）を持つ配列
    output = result["Plus214_Output_0"]
    output.is_a?(Array(Float32)).should be_true
    if output.is_a?(Array(Float32))
      output.size.should eq(10)
    end
  end
  
  it "predict with different data types" do
    model = OnnxRuntime::Model.new("spec/fixtures/mnist.onnx")
    
    # Float32型の入力データ
    input_float32 = Array(Float32).new(1 * 1 * 28 * 28, 0.0_f32)
    result_float32 = model.predict({"Input3" => input_float32}, nil, shape: {"Input3" => [1_i64, 1_i64, 28_i64, 28_i64]})
    result_float32["Plus214_Output_0"].is_a?(Array(Float32)).should be_true
    
    # Int32型の入力データ（自動的にFloat32に変換される）
    input_int32 = Array(Int32).new(1 * 1 * 28 * 28, 0)
    result_int32 = model.predict({"Input3" => input_int32}, nil, shape: {"Input3" => [1_i64, 1_i64, 28_i64, 28_i64]})
    result_int32["Plus214_Output_0"].is_a?(Array(Float32)).should be_true
  end
  
  it "metadata" do
    model = OnnxRuntime::Model.new("spec/fixtures/mnist.onnx")
    metadata = model.metadata
    metadata.should_not be_nil
  end
end

describe OnnxRuntime::SparseTensor do
  it "creates COO format sparse tensor" do
    # 2x3の疎行列を作成
    # [[1, 0, 2],
    #  [0, 3, 0]]
    values = [1.0_f32, 2.0_f32, 3.0_f32]
    indices = [[0, 0], [0, 2], [1, 1]]
    dense_shape = [2_i64, 3_i64]
    
    # COO形式のスパーステンソルを作成
    sparse_tensor = OnnxRuntime::SparseTensor(Float32).coo(values, indices.flatten, dense_shape)
    
    sparse_tensor.should be_a(OnnxRuntime::SparseTensor(Float32))
    sparse_tensor.values.should eq(values)
    sparse_tensor.dense_shape.should eq(dense_shape)
    sparse_tensor.format.should eq(OnnxRuntime::LibOnnxRuntime::SparseFormat::COO)
  end
  
  it "creates CSR format sparse tensor" do
    # 2x3の疎行列を作成
    # [[1, 0, 2],
    #  [0, 3, 0]]
    values = [1.0_f32, 2.0_f32, 3.0_f32]
    inner_indices = [0_i64, 2_i64, 1_i64]  # 列インデックス
    outer_indices = [0_i64, 2_i64, 3_i64]  # 行ポインタ
    dense_shape = [2_i64, 3_i64]
    
    # CSR形式のスパーステンソルを作成
    sparse_tensor = OnnxRuntime::SparseTensor(Float32).csr(values, inner_indices, outer_indices, dense_shape)
    
    sparse_tensor.should be_a(OnnxRuntime::SparseTensor(Float32))
    sparse_tensor.values.should eq(values)
    sparse_tensor.dense_shape.should eq(dense_shape)
    sparse_tensor.format.should eq(OnnxRuntime::LibOnnxRuntime::SparseFormat::CSRC)
  end
  
  it "creates sparse tensor with different data types" do
    # Float32型のスパーステンソル
    values_float32 = [1.0_f32, 2.0_f32, 3.0_f32]
    indices = [[0, 0], [0, 2], [1, 1]]
    dense_shape = [2_i64, 3_i64]
    
    sparse_tensor_float32 = OnnxRuntime::SparseTensor(Float32).coo(values_float32, indices.flatten, dense_shape)
    sparse_tensor_float32.should be_a(OnnxRuntime::SparseTensorFloat32)
    
    # Int32型のスパーステンソル
    values_int32 = [1, 2, 3]
    sparse_tensor_int32 = OnnxRuntime::SparseTensor(Int32).coo(values_int32, indices.flatten, dense_shape)
    sparse_tensor_int32.should be_a(OnnxRuntime::SparseTensorInt32)
    
    # Int64型のスパーステンソル
    values_int64 = [1_i64, 2_i64, 3_i64]
    sparse_tensor_int64 = OnnxRuntime::SparseTensor(Int64).coo(values_int64, indices.flatten, dense_shape)
    sparse_tensor_int64.should be_a(OnnxRuntime::SparseTensorInt64)
  end
end

describe OnnxRuntime do
  it "provides helper methods for creating sparse tensors" do
    values = [1.0_f32, 2.0_f32, 3.0_f32]
    indices = [[0, 0], [0, 2], [1, 1]]
    dense_shape = [2_i64, 3_i64]
    
    # モジュールレベルのヘルパーメソッドを使用
    sparse_tensor = OnnxRuntime.coo_sparse_tensor(values, indices.flatten, dense_shape)
    sparse_tensor.should be_a(OnnxRuntime::SparseTensor(Float32))
    
    inner_indices = [0_i64, 2_i64, 1_i64]
    outer_indices = [0_i64, 2_i64, 3_i64]
    csr_tensor = OnnxRuntime.csr_sparse_tensor(values, inner_indices, outer_indices, dense_shape)
    csr_tensor.should be_a(OnnxRuntime::SparseTensor(Float32))
  end
end
