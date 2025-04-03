module OnnxRuntime
  # SparseTensor class provides high-level API for working with sparse tensors in ONNX Runtime.
  class SparseTensor(T)
    getter :format, :values, :indices, :dense_shape

    # Creates a new SparseTensor instance.
    #
    # * `format` - The sparse tensor format (COO, CSR, or BLOCK_SPARSE)
    # * `values` - The non-zero values in the sparse tensor
    # * `indices` - The indices data for the sparse tensor (format-specific)
    # * `dense_shape` - The shape of the dense tensor this sparse tensor represents
    def initialize(@format : LibOnnxRuntime::SparseFormat, @values : Array(T), @indices : Hash(Symbol, Array(Int64) | Array(Int32)), @dense_shape : Array(Int64))
    end

    # Creates a COO format sparse tensor.
    #
    # * `values` - The non-zero values in the sparse tensor
    # * `indices` - The indices data for the sparse tensor (2D array where each row is a coordinate)
    # * `dense_shape` - The shape of the dense tensor this sparse tensor represents
    def self.coo(values : Array(T), indices : Array, dense_shape : Array(Int64)) forall T
      # For COO format, indices should be a 2D array where each row is a coordinate
      # Convert to the format expected by the SparseTensor constructor
      indices_hash = {:coo_indices => indices.flatten.map(&.to_i64)}

      new(LibOnnxRuntime::SparseFormat::COO, values, indices_hash, dense_shape)
    end

    # Creates a CSR format sparse tensor.
    #
    # * `values` - The non-zero values in the sparse tensor
    # * `inner_indices` - The column indices for each non-zero value
    # * `outer_indices` - The row pointers indicating where each row starts in the values array
    # * `dense_shape` - The shape of the dense tensor this sparse tensor represents
    def self.csr(values : Array(T), inner_indices : Array, outer_indices : Array, dense_shape : Array(Int64)) forall T
      indices_hash = {
        :inner_indices => inner_indices.map(&.to_i64),
        :outer_indices => outer_indices.map(&.to_i64),
      }

      new(LibOnnxRuntime::SparseFormat::CSRC, values, indices_hash, dense_shape)
    end

    # Creates a BlockSparse format sparse tensor.
    #
    # * `values` - The non-zero values in the sparse tensor
    # * `indices` - The block indices
    # * `dense_shape` - The shape of the dense tensor this sparse tensor represents
    def self.block_sparse(values : Array(T), indices : Array, dense_shape : Array(Int64)) forall T
      indices_hash = {:block_indices => indices.map(&.to_i32)}

      new(LibOnnxRuntime::SparseFormat::BLOCK_SPARSE, values, indices_hash, dense_shape)
    end

    # Converts the sparse tensor to an OrtValue that can be used with the ONNX Runtime API.
    #
    # * `session` - The InferenceSession instance to use for creating the OrtValue
    def to_ort_value(session)
      api = session.api
      allocator = session.allocator

      # Create a sparse tensor OrtValue
      tensor = Pointer(LibOnnxRuntime::OrtValue).null
      status = api.create_sparse_tensor_as_ort_value.call(
        allocator,
        @dense_shape.to_unsafe,
        @dense_shape.size.to_u64,
        element_data_type,
        pointerof(tensor)
      )
      session.check_status(status)

      # Create memory info for CPU
      memory_info = Pointer(LibOnnxRuntime::OrtMemoryInfo).null
      status = api.create_cpu_memory_info.call(
        LibOnnxRuntime::AllocatorType::DEVICE,
        LibOnnxRuntime::MemType::CPU,
        pointerof(memory_info)
      )
      session.check_status(status)

      # Fill the sparse tensor with values and indices based on the format
      case @format
      when LibOnnxRuntime::SparseFormat::COO
        fill_coo_tensor(api, tensor, memory_info, session)
      when LibOnnxRuntime::SparseFormat::CSRC
        fill_csr_tensor(api, tensor, memory_info, session)
      when LibOnnxRuntime::SparseFormat::BLOCK_SPARSE
        fill_block_sparse_tensor(api, tensor, memory_info, session)
      else
        raise "Unsupported sparse format: #{@format}"
      end

      api.release_memory_info.call(memory_info)

      tensor
    end


    private def element_data_type
      case @values
      when Array(Float32)
        LibOnnxRuntime::TensorElementDataType::FLOAT
      when Array(Int32)
        LibOnnxRuntime::TensorElementDataType::INT32
      when Array(Int64)
        LibOnnxRuntime::TensorElementDataType::INT64
      when Array(Float64)
        LibOnnxRuntime::TensorElementDataType::DOUBLE
      when Array(UInt8)
        LibOnnxRuntime::TensorElementDataType::UINT8
      when Array(Int8)
        LibOnnxRuntime::TensorElementDataType::INT8
      when Array(UInt16)
        LibOnnxRuntime::TensorElementDataType::UINT16
      when Array(Int16)
        LibOnnxRuntime::TensorElementDataType::INT16
      when Array(UInt32)
        LibOnnxRuntime::TensorElementDataType::UINT32
      when Array(UInt64)
        LibOnnxRuntime::TensorElementDataType::UINT64
      when Array(Bool)
        LibOnnxRuntime::TensorElementDataType::BOOL
      else
        raise "Unsupported value type: #{@values.class}"
      end
    end

    private def fill_coo_tensor(api, tensor, memory_info, session)
      # Prepare values shape (just the number of non-zero values)
      values_shape = [values.size.to_i64]

      # Get COO indices
      coo_indices = @indices[:coo_indices].as(Array(Int64))

      # Fill the sparse tensor with COO format data based on value type
      values_ptr = Pointer(Void).null
      case @values
      when Array(Float32)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      when Array(Int32)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      when Array(Int64)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      when Array(Float64)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      else
        raise "Unsupported value type: #{@values.class}"
      end

      status = api.fill_sparse_tensor_coo.call(
        tensor,
        memory_info,
        values_shape.to_unsafe,
        values_shape.size.to_u64,
        values_ptr,
        coo_indices.to_unsafe,
        coo_indices.size.to_u64
      )
      session.check_status(status)
    end

    private def fill_csr_tensor(api, tensor, memory_info, session)
      # Prepare values shape (just the number of non-zero values)
      values_shape = [values.size.to_i64]

      # Get CSR indices
      inner_indices = @indices[:inner_indices].as(Array(Int64))
      outer_indices = @indices[:outer_indices].as(Array(Int64))

      # Fill the sparse tensor with CSR format data based on value type
      values_ptr = Pointer(Void).null
      case @values
      when Array(Float32)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      when Array(Int32)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      when Array(Int64)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      when Array(Float64)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      else
        raise "Unsupported value type: #{@values.class}"
      end

      status = api.fill_sparse_tensor_csr.call(
        tensor,
        memory_info,
        values_shape.to_unsafe,
        values_shape.size.to_u64,
        values_ptr,
        inner_indices.to_unsafe,
        inner_indices.size.to_u64,
        outer_indices.to_unsafe,
        outer_indices.size.to_u64
      )
      session.check_status(status)
    end

    private def fill_block_sparse_tensor(api, tensor, memory_info, session)
      # Prepare values shape (just the number of non-zero values)
      values_shape = [values.size.to_i64]

      # Get block indices
      block_indices = @indices[:block_indices].as(Array(Int32))

      # For BlockSparse format, we need the shape of the indices
      indices_shape = [block_indices.size.to_i64]

      # Fill the sparse tensor with BlockSparse format data based on value type
      values_ptr = Pointer(Void).null
      case @values
      when Array(Float32)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      when Array(Int32)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      when Array(Int64)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      when Array(Float64)
        ptr = @values.to_unsafe
        values_ptr = Pointer(Void).new(ptr.address)
      else
        raise "Unsupported value type: #{@values.class}"
      end

      status = api.fill_sparse_tensor_block_sparse.call(
        tensor,
        memory_info,
        values_shape.to_unsafe,
        values_shape.size.to_u64,
        values_ptr,
        indices_shape.to_unsafe,
        indices_shape.size.to_u64,
        block_indices.to_unsafe
      )
      session.check_status(status)
    end
  end
end
