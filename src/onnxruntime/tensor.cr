module OnnxRuntime
  # Type alias for named tensors (output from inference)
  alias NamedTensors = Hash(String, Array(Float32) | Array(Float64) | Array(Int32) | Array(Int64) | Array(Bool) | Array(String) | Array(UInt8) | Array(Int8) | Array(UInt16) | Array(Int16) | Array(UInt32) | Array(UInt64) | SparseTensorFloat32 | SparseTensorInt32 | SparseTensorInt64 | SparseTensorFloat64)

  # Tensor class provides high-level API for working with dense tensors in ONNX Runtime.
  class Tensor
    # Creates a tensor from various data types
    def self.create(data : Array(Float32), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::FLOAT, sizeof(Float32), session)
    end

    def self.create(data : Array(Int32), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::INT32, sizeof(Int32), session)
    end

    def self.create(data : Array(Int64), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::INT64, sizeof(Int64), session)
    end

    def self.create(data : Array(Float64), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::DOUBLE, sizeof(Float64), session)
    end

    def self.create(data : Array(UInt8), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::UINT8, sizeof(UInt8), session)
    end

    def self.create(data : Array(Int8), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::INT8, sizeof(Int8), session)
    end

    def self.create(data : Array(UInt16), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::UINT16, sizeof(UInt16), session)
    end

    def self.create(data : Array(Int16), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::INT16, sizeof(Int16), session)
    end

    def self.create(data : Array(UInt32), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::UINT32, sizeof(UInt32), session)
    end

    def self.create(data : Array(UInt64), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::UINT64, sizeof(UInt64), session)
    end

    def self.create(data : Array(Bool), shape = nil, session = nil)
      create_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::BOOL, sizeof(Bool), session)
    end

    def self.create(data : Array(String), shape = nil, session = nil)
      create_string_tensor(data, shape, LibOnnxRuntime::TensorElementDataType::STRING, session)
    end

    def self.create(data, shape = nil, session = nil)
      raise "Unsupported data type: #{data.class}, #{data.inspect}, shape: #{shape.inspect}"
    end

    # Create a tensor with data
    private def self.create_with_data(data, shape, element_type, element_size, session)
      shape = [data.size.to_i64] if shape.nil?
      tensor = Pointer(LibOnnxRuntime::OrtValue).null
      
      # Get session and allocator
      if session.nil?
        # Session is required for creating tensors
        raise "Session is required for creating tensors"
      end
      
      api = session.api
      allocator = session.allocator
      memory_info = create_cpu_memory_info(api)

      status = api.create_tensor_with_data_as_ort_value.call(
        memory_info,
        data.to_unsafe.as(Void*),
        (data.size * element_size).to_u64,
        shape.to_unsafe,
        shape.size.to_u64,
        element_type,
        pointerof(tensor)
      )
      
      session.check_status(status)
      tensor
    ensure
      api.release_memory_info.call(memory_info) if memory_info && api
    end

    # Create a string tensor
    private def self.create_string_tensor(data, shape, element_type, session)
      shape = [data.size.to_i64, 1_i64] if shape.nil?
      shape = shape.map { |val| val < 0 ? data.size.to_i64 : val }
      tensor = Pointer(LibOnnxRuntime::OrtValue).null
      
      # Get session and allocator
      if session.nil?
        # Session is required for creating tensors
        raise "Session is required for creating tensors"
      end
      
      api = session.api
      allocator = session.allocator

      status = api.create_tensor_as_ort_value.call(
        allocator,
        shape.to_unsafe,
        shape.size.to_u64,
        element_type,
        pointerof(tensor)
      )
      
      session.check_status(status)

      cstrs = data.map(&.to_unsafe)
      str_ptrs = Pointer(UInt8*).malloc(data.size)
      data.each_with_index { |_, i| str_ptrs[i] = cstrs[i] }

      status = api.fill_string_tensor.call(
        tensor,
        str_ptrs,
        data.size.to_u64
      )
      
      session.check_status(status)
      tensor
    end

    # Helper method to create CPU memory info
    private def self.create_cpu_memory_info(api)
      memory_info = Pointer(LibOnnxRuntime::OrtMemoryInfo).null
      status = api.create_cpu_memory_info.call(
        LibOnnxRuntime::AllocatorType::DEVICE,
        LibOnnxRuntime::MemType::CPU,
        pointerof(memory_info)
      )
      check_status(status, api)
      memory_info
    end

    # Helper method to check status
    private def self.check_status(status, api)
      return if status.null?

      error_code = api.get_error_code.call(status)
      error_message = String.new(api.get_error_message.call(status))
      api.release_status.call(status)

      raise "ONNXRuntime Error: #{error_message} (#{error_code})"
    end

    # Extract data from a tensor
    def self.extract_data(tensor, session)
      # Get type info
      type_info = get_type_info(tensor, session)
      api = session.api

      onnx_type = get_onnx_type_from_type_info(type_info, session)

      case onnx_type
      when LibOnnxRuntime::OnnxType::TENSOR
        # Handle dense tensor
        extract_dense_tensor_data(tensor, type_info, session)
      when LibOnnxRuntime::OnnxType::SPARSETENSOR
        # Handle sparse tensor
        extract_sparse_tensor_data(tensor, session)
      else
        raise "Unsupported ONNX type: #{onnx_type}"
      end
    ensure
      api.release_type_info.call(type_info) if type_info && api
    end

    # Extract data from a dense tensor
    private def self.extract_dense_tensor_data(tensor, type_info, session)
      api = session.api

      tensor_info = cast_type_info_to_tensor_info(type_info, session)
      element_type = get_tensor_element_type(tensor_info, session)
      dims_count = get_dimensions_count(tensor_info, session)
      dims = get_dimensions(tensor_info, dims_count, session)
      element_count = calculate_total(dims)
      data_ptr = get_tensor_mutable_data(tensor, session)

      data_ptr_to_data(data_ptr, element_type, element_count)
    end

    # Convert data pointer to Crystal array
    private def self.data_ptr_to_data(data_ptr, element_type, element_count)
      case element_type
      when .float?
        Slice.new(data_ptr.as(Float32*), element_count)
      when .int32?
        Slice.new(data_ptr.as(Int32*), element_count)
      when .int64?
        Slice.new(data_ptr.as(Int64*), element_count)
      when .double?
        Slice.new(data_ptr.as(Float64*), element_count)
      when .uint8?
        Slice.new(data_ptr.as(UInt8*), element_count)
      when .int8?
        Slice.new(data_ptr.as(Int8*), element_count)
      when .uint16?
        Slice.new(data_ptr.as(UInt16*), element_count)
      when .int16?
        Slice.new(data_ptr.as(Int16*), element_count)
      when .uint32?
        Slice.new(data_ptr.as(UInt32*), element_count)
      when .uint64?
        Slice.new(data_ptr.as(UInt64*), element_count)
      when .bool?
        Slice.new(data_ptr.as(Bool*), element_count)
      else
        raise "Unsupported tensor element type: #{element_type}"
      end
        .to_a
    end

    # Helper methods for tensor operations
    private def self.get_type_info(tensor, session)
      type_info = Pointer(LibOnnxRuntime::OrtTypeInfo).null
      status = session.api.get_type_info.call(tensor, pointerof(type_info))
      session.check_status(status)
      type_info
    end

    private def self.get_onnx_type_from_type_info(type_info, session)
      onnx_type = LibOnnxRuntime::OnnxType::TENSOR
      status = session.api.get_onnx_type_from_type_info.call(type_info, pointerof(onnx_type))
      session.check_status(status)
      onnx_type
    end

    private def self.cast_type_info_to_tensor_info(type_info, session)
      tensor_info = Pointer(LibOnnxRuntime::OrtTensorTypeAndShapeInfo).null
      status = session.api.cast_type_info_to_tensor_info.call(type_info, pointerof(tensor_info))
      session.check_status(status)
      tensor_info
    end

    private def self.get_tensor_element_type(tensor_info, session)
      element_type = LibOnnxRuntime::TensorElementDataType::FLOAT
      status = session.api.get_tensor_element_type.call(tensor_info, pointerof(element_type))
      session.check_status(status)
      TensorElementDataType.new(element_type)
    end

    private def self.get_dimensions_count(tensor_info, session)
      dims_count = 0_u64
      status = session.api.get_dimensions_count.call(tensor_info, pointerof(dims_count))
      session.check_status(status)
      dims_count
    end

    private def self.get_dimensions(tensor_info, dims_count, session)
      dims = Array(Int64).new(dims_count, 0_i64)
      status = session.api.get_dimensions.call(tensor_info, dims.to_unsafe, dims_count)
      session.check_status(status)
      dims
    end

    private def self.get_tensor_mutable_data(tensor, session)
      data_ptr = Pointer(Void).null
      status = session.api.get_tensor_mutable_data.call(tensor, pointerof(data_ptr))
      session.check_status(status)
      data_ptr
    end

    private def self.calculate_total(dims, initial_value = 1_i64)
      # ameba:disable Lint/UselessAssign
      dims.reduce(initial_value) { |acc, dim| acc *= dim }
    end

    # Extract data from a sparse tensor
    private def self.extract_sparse_tensor_data(tensor, session)
      api = session.api

      # Get sparse tensor format
      format = get_sparse_tensor_format(tensor, session)

      # Get values type and shape
      values_info = get_sparse_tensor_values_type_and_shape(tensor, session)

      # Get element type
      element_type = get_tensor_element_type(values_info, session)

      # Get values shape
      dims_count = get_dimensions_count(values_info, session)
      values_shape = get_dimensions(values_info, dims_count, session)

      # Get values data
      data_ptr = get_sparse_tensor_values(tensor, session)

      # Calculate total number of values
      values_count = calculate_total(values_shape)

      # Extract values based on element type
      values = data_ptr_to_data(data_ptr, element_type, values_count)

      # Extract indices based on format
      indices = extract_indices_format(tensor, format, session)

      # Get dense shape from the first output
      dense_shape = session.outputs.first.shape

      # Create and return SparseTensor with the appropriate type
      case values
      when Array(Float32)
        SparseTensor(Float32).new(format, values, indices, dense_shape)
      when Array(Int32)
        SparseTensor(Int32).new(format, values, indices, dense_shape)
      when Array(Int64)
        SparseTensor(Int64).new(format, values, indices, dense_shape)
      when Array(Float64)
        SparseTensor(Float64).new(format, values, indices, dense_shape)
      else
        raise "Unsupported sparse tensor value type: #{values.class}"
      end
    end

    private def self.get_sparse_tensor_format(tensor, session)
      format = LibOnnxRuntime::SparseFormat::UNDEFINED
      status = session.api.get_sparse_tensor_format.call(tensor, pointerof(format))
      session.check_status(status)
      format
    end

    private def self.get_sparse_tensor_values_type_and_shape(tensor, session)
      values_info = Pointer(LibOnnxRuntime::OrtTensorTypeAndShapeInfo).null
      status = session.api.get_sparse_tensor_values_type_and_shape.call(tensor, pointerof(values_info))
      session.check_status(status)
      values_info
    end

    private def self.get_sparse_tensor_values(tensor, session)
      values_ptr = Pointer(Void).null
      status = session.api.get_sparse_tensor_values.call(tensor, pointerof(values_ptr))
      session.check_status(status)
      values_ptr
    end

    private def self.extract_indices_format(tensor, format, session)
      case format
      when LibOnnxRuntime::SparseFormat::COO
        {
          coo_indices: extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::COO_INDICES, session),
        }
      when LibOnnxRuntime::SparseFormat::CSRC
        {
          inner_indices: extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::CSR_INNER_INDICES, session),
          outer_indices: extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::CSR_OUTER_INDICES, session),
        }
      when LibOnnxRuntime::SparseFormat::BLOCK_SPARSE
        {
          block_indices: extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::BLOCK_SPARSE_INDICES, session),
        }
      else
        raise "Unsupported sparse format: #{format}"
      end
        .to_h
    end

    private def self.extract_sparse_indices(tensor, indices_format, session)
      api = session.api
      indices_info = Pointer(LibOnnxRuntime::OrtTensorTypeAndShapeInfo).null

      # Get indices type and shape
      status = api.get_sparse_tensor_indices_type_shape.call(tensor, indices_format, pointerof(indices_info))
      session.check_status(status)

      # Get indices shape
      dims_count = 0_u64
      status = api.get_dimensions_count.call(indices_info, pointerof(dims_count))
      session.check_status(status)

      indices_shape = Array(Int64).new(dims_count, 0_i64)
      status = api.get_dimensions.call(indices_info, indices_shape.to_unsafe, dims_count)
      session.check_status(status)

      # Get indices data
      indices_count = 0_u64
      indices_ptr = Pointer(Void).null
      status = api.get_sparse_tensor_indices.call(tensor, indices_format, pointerof(indices_count), pointerof(indices_ptr))
      session.check_status(status)

      data_ptr = indices_ptr

      # Calculate total number of indices
      total_indices = calculate_total(indices_shape)

      # Extract indices
      if indices_format == LibOnnxRuntime::SparseIndicesFormat::BLOCK_SPARSE_INDICES
        Slice.new(data_ptr.as(Int32*), total_indices).to_a
      else
        Slice.new(data_ptr.as(Int64*), total_indices).to_a
      end
    end
  end
end
