module OnnxRuntime
  class InferenceSession
    # Use OrtEnvironment singleton for environment management
    getter session : Pointer(LibOnnxRuntime::OrtSession)
    getter allocator : Pointer(LibOnnxRuntime::OrtAllocator)
    getter inputs : Array(TensorInfo)
    getter outputs : Array(TensorInfo)

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

    def initialize(path_or_bytes, provider : Provider? = nil, **session_options)
      # Initialize instance variables
      @inputs = [] of TensorInfo
      @outputs = [] of TensorInfo
      @session_released = true
      @allocator_released = true

      session_options_ptr = api_create_session_options

      # Set provider if provided
      if provider
        if provider_options = provider.options
          if provider_options.is_a?(CpuProviderOptions)
            # CPU provider is the default, no need to append
            # Do nothing
          elsif provider_options.is_a?(CudaProviderOptions)
            # For CUDA provider, use the V2 API
            status = api.session_options_append_execution_provider_cuda_v2.call(
              session_options_ptr,
              provider_options.to_unsafe
            )
            check_status(status)
          end
        end
      end

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
                 api.create_session.call(env, ort_path_string(path_or_bytes), session_options, pointerof(session))
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

    # Get input name at index
    private def api_get_input_name(session, index, allocator)
      name_ptr = Pointer(UInt8).null
      api_call &.session_get_input_name.call(session, index, allocator, pointerof(name_ptr))

      name_ptr.null? ? "" : copy_and_free_allocator_string(name_ptr, allocator)
    end

    # Get number of inputs
    private def api_get_input_count(session)
      count = 0_u64
      api_call &.session_get_input_count.call(session, pointerof(count))
      count
    end

    # Get input type info at index
    private def api_get_input_type_info(session, index)
      type_info = Pointer(LibOnnxRuntime::OrtTypeInfo).null
      api_call &.session_get_input_type_info.call(session, index, pointerof(type_info))
      type_info
    end

    # Get number of outputs
    private def api_get_output_count(session)
      count = 0_u64
      api_call &.session_get_output_count.call(session, pointerof(count))
      count
    end

    # Get output name at index
    private def api_get_output_name(session, index, allocator)
      name_ptr = Pointer(UInt8).null
      api_call &.session_get_output_name.call(session, index, allocator, pointerof(name_ptr))

      name_ptr.null? ? "" : copy_and_free_allocator_string(name_ptr, allocator)
    end

    # Copy allocator-owned ORT strings into managed Crystal strings and release the original buffer.
    private def copy_and_free_allocator_string(ptr, allocator)
      name = copy_c_string(ptr)
      api_call &.allocator_free.call(allocator, ptr.as(Void*))
      name
    end

    private def copy_c_string(ptr)
      length = LibC.strlen(ptr.as(LibC::Char*))
      raise "ORT string is too large to copy" if length > Int32::MAX
      bytesize = length.to_i32

      String.new(bytesize) do |buffer|
        buffer.copy_from(ptr, bytesize)
        {bytesize, bytesize}
      end
    end

    # Get output type info at index
    private def api_get_output_type_info(session, index)
      type_info = Pointer(LibOnnxRuntime::OrtTypeInfo).null
      api_call &.session_get_output_type_info.call(session, index, pointerof(type_info))
      type_info
    end

    # Load input tensor info
    private def load_inputs
      inputs = Array(TensorInfo).new
      count = api_get_input_count(@session)
      count.times do |i|
        name = api_get_input_name(@session, i, @allocator)
        type_info = api_get_input_type_info(@session, i)
        inputs << TensorInfo.from_type_info(name, type_info, self)
        api.release_type_info.call(type_info)
      end
      inputs
    end

    # Load output tensor info
    private def load_outputs
      outputs = Array(TensorInfo).new
      count = api_get_output_count(@session)
      count.times do |i|
        name = api_get_output_name(@session, i, @allocator)
        type_info = api_get_output_type_info(@session, i)
        outputs << TensorInfo.from_type_info(name, type_info, self)
        api.release_type_info.call(type_info)
      end
      outputs
    end

    def run(input_feed, output_names = nil, run_options : RunOptions? = nil, **options)
      # Track if we created run_options_ptr locally
      owned_run_options = run_options.nil?
      run_options_ptr = run_options ? run_options.to_unsafe : api_create_run_options
      input_tensors = [] of Pointer(LibOnnxRuntime::OrtValue)

      # Set run options if provided via hash
      if tag = options["tag"]?
        api_call(&.run_options_set_run_tag.call(run_options_ptr, tag.to_s))
      end

      if level = options["log_severity_level"]?
        api_call(&.run_options_set_run_log_severity_level.call(run_options_ptr, level.to_i))
      end

      if level = options["log_verbosity_level"]?
        api_call(&.run_options_set_run_log_verbosity_level.call(run_options_ptr, level.to_i))
      end

      # Prepare input tensors
      input_names = [] of String

      # Check if custom shapes are provided
      shapes = options["shape"]?.try &.as(Hash(String, Array(Int64)))

      input_feed.each do |name, data|
        shape = shapes.try &.[name]?
        tensor = data_to_tensor(name, data, shape)
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
          result[name] = Tensor.extract_data(tensor, self)
        end
      end
    ensure
      # Release output tensors that were allocated by the run() call
      output_tensors.each { |tensor| api.release_value.call(tensor) if tensor } if output_tensors

      # Clean up - only release if we created it
      api.release_run_options.call(run_options_ptr) if owned_run_options && run_options_ptr

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

    private def api_create_cpu_memory_info
      memory_info = Pointer(LibOnnxRuntime::OrtMemoryInfo).null
      api_call &.create_cpu_memory_info.call(LibOnnxRuntime::AllocatorType::DEVICE, LibOnnxRuntime::MemType::CPU, pointerof(memory_info))
      memory_info
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

    private def data_to_tensor(name, data, shape)
      if data.is_a?(SparseTensorFloat32 | SparseTensorInt32 | SparseTensorInt64 | SparseTensorFloat64)
        data.to_ort_value(self)
      elsif shape && data.is_a?(Array)
        # Create tensor with custom shape
        Tensor.create(data, shape, self)
      else
        Tensor.create(data, nil, self)
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

    private def ort_path_string(str : String)
      {% if flag?(:win32) %}
        utf16 = str.to_utf16
        utf16.to_unsafe.as(OnnxRuntime::LibOnnxRuntime::ORTCHAR_T*)
      {% else %}
        str.to_unsafe
      {% end %}
    end

    private def ort_string(str)
      str.to_unsafe
    end

    # Get the environment from the OrtEnvironment singleton
    private def env
      OrtEnvironment.instance.env
    end

    # Get model metadata
    def metadata
      ModelMetadata.from_session(self)
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
