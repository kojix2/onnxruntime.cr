require "./onnxruntime/libonnxruntime"
require "./onnxruntime/sparse_tensor"
require "./onnxruntime/inference_session"
require "./onnxruntime/model"
require "./onnxruntime/version"

module OnnxRuntime
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
