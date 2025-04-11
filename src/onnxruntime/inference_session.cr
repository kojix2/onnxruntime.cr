module OnnxRuntime
  class InputOutput
    getter name : String
    getter type : LibOnnxRuntime::TensorElementDataType
    getter shape : Array(Int64)

    def initialize(@name, @type, @shape)
    end
  end

  class InferenceSession
    # Use OrtEnvironment singleton for environment management
    getter session : Pointer(LibOnnxRuntime::OrtSession)
    getter allocator : Pointer(LibOnnxRuntime::OrtAllocator)
    getter inputs : Array(InputOutput)
    getter outputs : Array(InputOutput)

    @session_released = true   # Track if session has been released
    @allocator_released = true # Track if allocator has been released

    protected getter api : LibOnnxRuntime::Api { create_api }

    def create_api
      LibOnnxRuntime
        .OrtGetApiBase.value
        .get_api
        .call(OnnxRuntime::LibOnnxRuntime::ORT_API_VERSION)
        .value
    end

    def initialize(path_or_bytes, **session_options)
      session_options_ptr = api_create_session_options

      @session = load_session(path_or_bytes, session_options_ptr)
      @session_released = false
      @allocator = api_get_allocator_with_default_options
      @allocator_released = false
      @inputs = load_inputs
      @outputs = load_outputs
    ensure
      # Release session options
      api.release_session_options.call(session_options_ptr) if session_options_ptr
    end

    # Finalizer only releases session-specific resources
    # Environment is managed separately and should be released explicitly by the user
    def finalize
      release_session
      release_allocator
      # Note: Environment is not released here to avoid potential issues with concurrent finalization
    end

    # Method to explicitly release the session
    def release_session
      return if @session_released
      api.release_session.call(@session) if @session
      @session_released = true
    end

    # Method to explicitly release the allocator
    # Note: Default allocator may not need to be released, so commented out
    def release_allocator
      # return if @allocator_released
      # api.release_allocator.call(@allocator) if @allocator
      # @allocator_released = true
    end

    private def load_session(path_or_bytes, session_options)
      session = Pointer(LibOnnxRuntime::OrtSession).null
      status = if path_or_bytes.is_a?(String)
                 api.create_session.call(env, ort_string(path_or_bytes), session_options, pointerof(session))
               else
                 api.create_session_from_array.call(env, path_or_bytes.to_unsafe, path_or_bytes.size, session_options, pointerof(session))
               end
      check_status(status)
      session
    end

    private def api_get_allocator_with_default_options
      allocator = Pointer(LibOnnxRuntime::OrtAllocator).null
      api_call &.get_allocator_with_default_options.call(pointerof(allocator))
      allocator
    end

    private def api_session_get_input_name(session, index, allocator)
      name_ptr = Pointer(Pointer(UInt8)).malloc(1)
      api_call &.session_get_input_name.call(session, index, allocator, name_ptr)
      name_ptr
    end

    private def api_get_input_count(session)
      count = 0_u64
      api_call &.session_get_input_count.call(session, pointerof(count))
      count
    end

    private def api_session_get_input_type_info(session, index)
      type_info = Pointer(LibOnnxRuntime::OrtTypeInfo).null
      api_call &.session_get_input_type_info.call(session, index, pointerof(type_info))
      type_info
    end

    private def api_get_output_count(session)
      count = 0_u64
      api_call &.session_get_output_count.call(session, pointerof(count))
      count
    end

    private def api_session_get_output_name(session, index, allocator)
      name_ptr = Pointer(Pointer(UInt8)).malloc(1)
      api_call &.session_get_output_name.call(session, index, allocator, name_ptr)
      name_ptr
    end

    private def api_session_get_output_type_info(session, index)
      type_info = Pointer(LibOnnxRuntime::OrtTypeInfo).null
      api_call &.session_get_output_type_info.call(session, index, pointerof(type_info))
      type_info
    end

    private def load_inputs
      inputs = Array(InputOutput).new
      count = api_get_input_count(@session)
      count.times do |i|
        name_ptr = api_session_get_input_name(@session, i, @allocator)
        name = String.new(name_ptr.value)
        type_info = api_session_get_input_type_info(@session, i)
        inputs << type_info_to_input_output(name, type_info)
        api.release_type_info.call(type_info)
      end
      inputs
    end

    private def load_outputs
      outputs = Array(InputOutput).new
      count = api_get_output_count(@session)
      count.times do |i|
        name_ptr = api_session_get_output_name(@session, i, @allocator)
        name = String.new(name_ptr.value)
        type_info = api_session_get_output_type_info(@session, i)
        outputs << type_info_to_input_output(name, type_info)
        api.release_type_info.call(type_info)
      end
      outputs
    end

    def type_info_to_input_output(name, type_info)
      onnx_type = api_get_onnx_type_from_type_info(type_info)

      case onnx_type
      when LibOnnxRuntime::OnnxType::TENSOR
        tensor_info = api_cast_type_info_to_tensor_info(type_info)
        element_type = api_get_tensor_element_type(tensor_info)
        dims_count = api_get_dimensions_count(tensor_info)
        dims = api_get_dimensions(tensor_info, dims_count)
        InputOutput.new(name, LibOnnxRuntime::TensorElementDataType.new(element_type), dims)
      else
        raise "Unsupported ONNX type: #{onnx_type}"
      end
    end

    def run(input_feed, output_names = nil, **run_options)
      run_options_ptr = api_create_run_options
      input_tensors = [] of Pointer(LibOnnxRuntime::OrtValue)

      # Set run options if provided
      if tag = run_options["tag"]?
        api_call(&.run_options_set_run_tag.call(run_options_ptr, tag.to_s))
      end

      if level = run_options["log_severity_level"]?
        api_call(&.run_options_set_run_log_severity_level.call(run_options_ptr, level.to_i))
      end

      if level = run_options["log_verbosity_level"]?
        api_call(&.run_options_set_run_log_verbosity_level.call(run_options_ptr, level.to_i))
      end

      # Prepare input tensors
      input_names = [] of String

      # Check if custom shapes are provided
      shapes = run_options["shape"]?.try &.as(Hash(String, Array(Int64)))

      input_feed.each do |name, data|
        tensor = data_to_tensor(name, data, shapes[name]?)
        input_tensors << tensor
        input_names << name
      end

      # Prepare output names
      output_names = output_names || @outputs.map(&.name)

      # Prepare output tensors
      output_tensors = Array(Pointer(LibOnnxRuntime::OrtValue)).new(output_names.size, Pointer(LibOnnxRuntime::OrtValue).null)

      # Run inference
      input_names_ptr = input_names.map { |name| ort_string(name) }
      output_names_ptr = output_names.map { |name| ort_string(name) }

      api_call &.run.call(
        @session,
        run_options_ptr,
        input_names_ptr.to_unsafe,
        input_tensors.to_unsafe,
        input_tensors.size.to_u64,
        output_names_ptr.to_unsafe,
        output_names.size.to_u64,
        output_tensors.to_unsafe
      )

      # Extract output data
      NamedTensors.new.tap do |result|
        output_names.each_with_index do |name, i|
          next unless tensor = output_tensors[i]
          result[name] = extract_output_data(tensor)
          api.release_value.call(tensor)
        end
      end
    ensure
      # Clean up
      api.release_run_options.call(run_options_ptr) if run_options_ptr

      # Release input tensors
      input_tensors.each { |tensor| api.release_value.call(tensor) if tensor } if input_tensors
    end

    private def api_call(&)
      status = yield api
      check_status(status)
      status
    end

    private def api_create_session_options
      session_options_ptr = Pointer(LibOnnxRuntime::OrtSessionOptions).null
      api_call &.create_session_options.call(pointerof(session_options_ptr))
      session_options_ptr
    end

    private def api_create_run_options
      run_options_ptr = Pointer(LibOnnxRuntime::OrtRunOptions).null
      api_call &.create_run_options.call(pointerof(run_options_ptr))
      run_options_ptr
    end

    private def api_cast_type_info_to_tensor_info(type_info)
      tensor_info = Pointer(LibOnnxRuntime::OrtTensorTypeAndShapeInfo).null
      api_call &.cast_type_info_to_tensor_info.call(type_info, pointerof(tensor_info))
      tensor_info
    end

    private def api_get_tensor_element_type(tensor_info)
      element_type = LibOnnxRuntime::TensorElementDataType::FLOAT
      api_call &.get_tensor_element_type.call(tensor_info, pointerof(element_type))
      TensorElementDataType.new(element_type)
    end

    private def api_get_dimensions_count(tensor_info)
      dims_count = 0_u64
      api_call &.get_dimensions_count.call(tensor_info, pointerof(dims_count))
      dims_count
    end

    private def api_get_dimensions(tensor_info, dims_count)
      dims = Array(Int64).new(dims_count, 0_i64)
      api_call &.get_dimensions.call(tensor_info, dims.to_unsafe, dims_count)
      dims
    end

    private def api_get_tensor_mutable_data(tensor)
      data_ptr = Pointer(Void).null
      api_call &.get_tensor_mutable_data.call(tensor, pointerof(data_ptr))
      data_ptr
    end

    private def api_get_type_info(tensor)
      type_info = Pointer(LibOnnxRuntime::OrtTypeInfo).null
      api_call &.get_type_info.call(tensor, pointerof(type_info))
      type_info
    end

    private def api_get_onnx_type_from_type_info(type_info)
      onnx_type = LibOnnxRuntime::OnnxType::TENSOR
      api_call &.get_onnx_type_from_type_info.call(type_info, pointerof(onnx_type))
      onnx_type
    end

    private def api_get_sparse_tensor_format(tensor)
      format = LibOnnxRuntime::SparseFormat::UNDEFINED
      api_call &.get_sparse_tensor_format.call(tensor, pointerof(format))
      format
    end

    private def api_get_sparse_tensor_values_type_and_shape(tensor)
      values_info = Pointer(LibOnnxRuntime::OrtTensorTypeAndShapeInfo).null
      api_call &.get_sparse_tensor_values_type_and_shape.call(tensor, pointerof(values_info))
      values_info
    end

    private def api_get_sparse_tensor_values(tensor)
      values_ptr = Pointer(Void).null
      api_call &.get_sparse_tensor_values.call(tensor, pointerof(values_ptr))
      values_ptr
    end

    private def data_to_tensor(name, data, shape)
      if data.is_a?(SparseTensorFloat32 | SparseTensorInt32 | SparseTensorInt64 | SparseTensorFloat64)
        data.to_ort_value(self)
      elsif shape && data.is_a?(Array)
        # Create tensor with custom shape
        create_tensor(data, shape)
      else
        create_tensor(data)
      end
    end

    # Extract data from the output tensor
    def extract_output_data(tensor)
      type_info = api_get_type_info(tensor)
      onnx_type = api_get_onnx_type_from_type_info(type_info)

      case onnx_type
      when LibOnnxRuntime::OnnxType::TENSOR
        # Handle dense tensor
        extract_dense_tensor_data(tensor, type_info)
      when LibOnnxRuntime::OnnxType::SPARSETENSOR
        # Handle sparse tensor
        extract_sparse_tensor_data(tensor)
      else
        raise "Unsupported ONNX type: #{onnx_type}"
      end
    ensure
      api.release_type_info.call(type_info) if type_info
    end

    # Extract data from a dense tensor
    private def extract_dense_tensor_data(tensor, type_info)
      tensor_info = api_cast_type_info_to_tensor_info(type_info)
      element_type = api_get_tensor_element_type(tensor_info)
      dims_count = api_get_dimensions_count(tensor_info)
      dims = api_get_dimensions(tensor_info, dims_count)
      element_count = calculate_total(dims)
      data_ptr = api_get_tensor_mutable_data(tensor)
      data_ptr_to_data(data_ptr, element_type, element_count)
    end

    private def data_ptr_to_data(data_ptr, element_type, element_count)
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

    private def extract_indices_format(tensor, format)
      case format
      when LibOnnxRuntime::SparseFormat::COO
        {
          coo_indices: extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::COO_INDICES),
        }
      when LibOnnxRuntime::SparseFormat::CSRC
        {
          inner_indices: extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::CSR_INNER_INDICES),
          outer_indices: extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::CSR_OUTER_INDICES),
        }
      when LibOnnxRuntime::SparseFormat::BLOCK_SPARSE
        {
          block_indices: extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::BLOCK_SPARSE_INDICES),
        }
      else
        raise "Unsupported sparse format: #{format}"
      end
        .to_h
    end

    # Extract data from a sparse tensor
    private def extract_sparse_tensor_data(tensor)
      # Get sparse tensor format
      format = api_get_sparse_tensor_format(tensor)

      # Get values type and shape
      values_info = api_get_sparse_tensor_values_type_and_shape(tensor)

      # Get element type
      element_type = api_get_tensor_element_type(values_info)

      # Get values shape
      dims_count = api_get_dimensions_count(values_info)

      values_shape = api_get_dimensions(values_info, dims_count)

      # Get values data
      data_ptr = api_get_sparse_tensor_values(tensor)

      # Calculate total number of values
      values_count = calculate_total(values_shape)

      # Extract values based on element type
      values = data_ptr_to_data(data_ptr, element_type, values_count)

      # Extract indices based on format
      indices = extract_indices_format(tensor, format)

      # Get dense shape from the first output
      dense_shape = @outputs.first.shape

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

    # Extract indices from a sparse tensor
    private def extract_sparse_indices(tensor, indices_format)
      indices_info = Pointer(LibOnnxRuntime::OrtTensorTypeAndShapeInfo).null

      # Get indices type and shape
      status = api.get_sparse_tensor_indices_type_shape.call(tensor, indices_format, pointerof(indices_info))
      check_status(status)

      # Get indices shape
      dims_count = 0_u64
      status = api.get_dimensions_count.call(indices_info, pointerof(dims_count))
      check_status(status)

      indices_shape = Array(Int64).new(dims_count, 0_i64)
      status = api.get_dimensions.call(indices_info, indices_shape.to_unsafe, dims_count)
      check_status(status)

      # Get indices data
      indices_count = 0_u64
      indices_ptr = Pointer(Void).null
      status = api.get_sparse_tensor_indices.call(tensor, indices_format, pointerof(indices_count), pointerof(indices_ptr))
      check_status(status)

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

    private def calculate_total(dims, initial_value = 1_i64)
      # ameba:disable Lint/UselessAssign
      dims.reduce(initial_value) { |acc, dim| acc *= dim }
    end

    private def create_tensor(data : Array(Float32), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::FLOAT, sizeof(Float32))
    end

    private def create_tensor(data : Array(Int32), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::INT32, sizeof(Int32))
    end

    private def create_tensor(data : Array(Int64), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::INT64, sizeof(Int64))
    end

    private def create_tensor(data : Array(Float64), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::DOUBLE, sizeof(Float64))
    end

    private def create_uint8_tensor(data : Array(UInt8), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::UINT8, sizeof(UInt8))
    end

    private def create_int8_tensor(data : Array(Int8), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::INT8, sizeof(Int8))
    end

    private def create_uint16_tensor(data : Array(UInt16), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::UINT16, sizeof(UInt16))
    end

    private def create_int16_tensor(data : Array(Int16), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::INT16, sizeof(Int16))
    end

    private def create_uint32_tensor(data : Array(UInt32), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::UINT32, sizeof(UInt32))
    end

    private def create_uint64_tensor(data : Array(UInt64), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::UINT64, sizeof(UInt64))
    end

    private def create_bool_tensor(data : Array(Bool), shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::BOOL, sizeof(Bool))
    end

    private def create_tensor(data, shape = nil)
      raise "Unsupported data type: #{data.class}"
    end

    private def create_tensor_with_data(data, shape, element_type, element_size)
      shape = [data.size.to_i64] if shape.nil?
      tensor = Pointer(LibOnnxRuntime::OrtValue).null
      memory_info = Pointer(LibOnnxRuntime::OrtMemoryInfo).null

      begin
        status = api.create_cpu_memory_info.call(LibOnnxRuntime::AllocatorType::DEVICE, LibOnnxRuntime::MemType::CPU, pointerof(memory_info))
        check_status(status)

        status = api.create_tensor_with_data_as_ort_value.call(
          memory_info,
          data.to_unsafe.as(Void*),
          (data.size * element_size).to_u64,
          shape.to_unsafe,
          shape.size.to_u64,
          element_type,
          pointerof(tensor)
        )
        check_status(status)

        tensor
      ensure
        api.release_memory_info.call(memory_info) if memory_info
      end
    end

    # Make check_status public so it can be used by SparseTensor
    def check_status(status)
      return if status.null?

      error_code = api.get_error_code.call(status)
      error_message = String.new(api.get_error_message.call(status))
      api.release_status.call(status)

      raise "ONNXRuntime Error: #{error_message} (#{error_code})"
    end

    private def ort_string(str)
      str.to_unsafe
    end

    # Get the environment from the OrtEnvironment singleton
    private def env
      OrtEnvironment.instance.env
    end

    # Class method to explicitly release the environment
    # This method is now a wrapper around OrtEnvironment.instance.release
    def self.release_env
      OrtEnvironment.instance.release
    rescue ex
      # Log error but don't propagate it during shutdown
      STDERR.puts "Error releasing ONNX Runtime environment: #{ex.message}"
    end
  end
end
