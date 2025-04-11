require "./onnxruntime/libonnxruntime"
require "./onnxruntime/sparse_tensor"
require "./onnxruntime/ort_environment"
require "./onnxruntime/inference_session"
require "./onnxruntime/model"
require "./onnxruntime/version"

module OnnxRuntime
  alias TensorType = Array(Float32) | Array(Float64) | Array(Int32) | Array(Int64) | Array(Bool) | Array(UInt8) | Array(Int8) | Array(UInt16) | Array(Int16) | Array(UInt32) | Array(UInt64) | SparseTensorFloat32 | SparseTensorInt32 | SparseTensorInt64 | SparseTensorFloat64

  alias NamedTensors = Hash(String, TensorType)

  alias TensorElementDataType = LibOnnxRuntime::TensorElementDataType

  # Create a COO format sparse tensor
  def self.coo_sparse_tensor(values, indices, dense_shape)
    SparseTensor.coo(values, indices, dense_shape)
  end

  # Create a CSR format sparse tensor
  def self.csr_sparse_tensor(values, inner_indices, outer_indices, dense_shape)
    SparseTensor.csr(values, inner_indices, outer_indices, dense_shape)
  end

  # Create a BlockSparse format sparse tensor
  def self.block_sparse_tensor(values, indices, dense_shape)
    SparseTensor.block_sparse(values, indices, dense_shape)
  end
end
