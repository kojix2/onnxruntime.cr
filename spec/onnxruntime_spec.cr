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
    session.inputs.should be_a(Array(OnnxRuntime::InputOutput))
    session.outputs.should be_a(Array(OnnxRuntime::InputOutput))
  end
end

describe OnnxRuntime::Model do
  it "create" do
    model = OnnxRuntime::Model.new("spec/fixtures/mnist.onnx")
    model.should be_a(OnnxRuntime::Model)
    model.inputs.should be_a(Array(OnnxRuntime::InputOutput))
    model.outputs.should be_a(Array(OnnxRuntime::InputOutput))
  end

  it "predict" do
    model = OnnxRuntime::Model.new("spec/fixtures/mnist.onnx")

    # MNIST model input is 4-dimensional [batch size, channels, height, width]
    # Create dummy input data (all zeros image)
    input_data = Array(Float32).new(1 * 1 * 28 * 28, 0.0_f32)

    # Execute prediction (specifying 4-dimensional shape)
    result = model.predict({"Input3" => input_data})

    # Result is a probability distribution for 10 classes (digits 0-9)
    result.should be_a(OnnxRuntime::NamedTensors)
    result.keys.should contain("Plus214_Output_0")

    # Output is an array with 10 values (probability for each digit)
    output = result["Plus214_Output_0"]
    output.is_a?(Array(Float32)).should be_true
    if output.is_a?(Array(Float32))
      output.size.should eq(10)
    end
  end

  it "predict with different data types" do
    model = OnnxRuntime::Model.new("spec/fixtures/mnist.onnx")

    # Float32 type input data
    input_float32 = Array(Float32).new(1 * 1 * 28 * 28, 0.0_f32)
    result_float32 = model.predict({"Input3" => input_float32})
    result_float32["Plus214_Output_0"].is_a?(Array(Float32)).should be_true

    # Int32 type input data (automatically converted to Float32)
    input_int32 = Array(Int32).new(1 * 1 * 28 * 28, 0)
    result_int32 = model.predict({"Input3" => input_int32})
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
    # Create a 2x3 sparse matrix
    # [[1, 0, 2],
    #  [0, 3, 0]]
    values = [1.0_f32, 2.0_f32, 3.0_f32]
    indices = [[0, 0], [0, 2], [1, 1]]
    dense_shape = [2_i64, 3_i64]

    # Create a sparse tensor in COO format
    sparse_tensor = OnnxRuntime::SparseTensor(Float32).coo(values, indices.flatten, dense_shape)

    sparse_tensor.should be_a(OnnxRuntime::SparseTensor(Float32))
    sparse_tensor.values.should eq(values)
    sparse_tensor.dense_shape.should eq(dense_shape)
    sparse_tensor.format.should eq(OnnxRuntime::LibOnnxRuntime::SparseFormat::COO)
  end

  it "creates CSR format sparse tensor" do
    # Create a 2x3 sparse matrix
    # [[1, 0, 2],
    #  [0, 3, 0]]
    values = [1.0_f32, 2.0_f32, 3.0_f32]
    inner_indices = [0_i64, 2_i64, 1_i64] # Column indices
    outer_indices = [0_i64, 2_i64, 3_i64] # Row pointers
    dense_shape = [2_i64, 3_i64]

    # Create a sparse tensor in CSR format
    sparse_tensor = OnnxRuntime::SparseTensor(Float32).csr(values, inner_indices, outer_indices, dense_shape)

    sparse_tensor.should be_a(OnnxRuntime::SparseTensor(Float32))
    sparse_tensor.values.should eq(values)
    sparse_tensor.dense_shape.should eq(dense_shape)
    sparse_tensor.format.should eq(OnnxRuntime::LibOnnxRuntime::SparseFormat::CSRC)
  end

  it "creates sparse tensor with different data types" do
    # Float32 type sparse tensor
    values_float32 = [1.0_f32, 2.0_f32, 3.0_f32]
    indices = [[0, 0], [0, 2], [1, 1]]
    dense_shape = [2_i64, 3_i64]

    sparse_tensor_float32 = OnnxRuntime::SparseTensor(Float32).coo(values_float32, indices.flatten, dense_shape)
    sparse_tensor_float32.should be_a(OnnxRuntime::SparseTensorFloat32)

    # Int32 type sparse tensor
    values_int32 = [1, 2, 3]
    sparse_tensor_int32 = OnnxRuntime::SparseTensor(Int32).coo(values_int32, indices.flatten, dense_shape)
    sparse_tensor_int32.should be_a(OnnxRuntime::SparseTensorInt32)

    # Int64 type sparse tensor
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

    # Use module level helper methods
    sparse_tensor = OnnxRuntime.coo_sparse_tensor(values, indices.flatten, dense_shape)
    sparse_tensor.should be_a(OnnxRuntime::SparseTensor(Float32))

    inner_indices = [0_i64, 2_i64, 1_i64]
    outer_indices = [0_i64, 2_i64, 3_i64]
    csr_tensor = OnnxRuntime.csr_sparse_tensor(values, inner_indices, outer_indices, dense_shape)
    csr_tensor.should be_a(OnnxRuntime::SparseTensor(Float32))
  end
end
