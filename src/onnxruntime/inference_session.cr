module OnnxRuntime
  class InferenceSession
    getter :inputs, :outputs, :session, :allocator

    def api
      OnnxRuntime::LibOnnxRuntime
        .OrtGetApiBase.value
        .get_api
        .call(OnnxRuntime::LibOnnxRuntime::ORT_API_VERSION)
        .value
    end

    def initialize(path_or_bytes, **session_options)
      session_options_ptr = api.create_session_options.call
      
      @session = load_session(path_or_bytes, session_options_ptr)
      @allocator = load_allocator
      @inputs = load_inputs
      @outputs = load_outputs
    end

    def finalize
      api.release_session.call(@session) if @session
    end

    private def load_session(path_or_bytes, session_options)
      session = Pointer(Void).null
      status = if path_or_bytes.is_a?(String)
                api.create_session.call(env, ort_string(path_or_bytes), session_options, pointerof(session))
              else
                api.create_session_from_array.call(env, path_or_bytes.to_unsafe, path_or_bytes.size, session_options, pointerof(session))
              end
      check_status(status)
      session
    end

    private def load_allocator
      allocator = Pointer(Void).null
      status = api.get_allocator_with_default_options.call(pointerof(allocator))
      check_status(status)
      allocator
    end

    private def load_inputs
      count = 0_u64
      status = api.session_get_input_count.call(@session, pointerof(count))
      check_status(status)

      inputs = [] of NamedTuple(name: String, type: LibOnnxRuntime::TensorElementDataType, shape: Array(Int64))
      
      count.times do |i|
        name_ptr = Pointer(Void).null
        name_len = 0_u64
        status = api.session_get_input_name.call(@session, i, name_ptr, pointerof(name_len))
        check_status(status)
        
        name_buffer = Bytes.new(name_len)
        status = api.session_get_input_name.call(@session, i, name_buffer.to_unsafe, pointerof(name_len))
        check_status(status)
        name = String.new(name_buffer)
        
        type_info = Pointer(Void).null
        status = api.session_get_input_type_info.call(@session, i, pointerof(type_info))
        check_status(status)
        
        onnx_type = 0
        status = api.get_onnx_type_from_type_info.call(type_info, pointerof(onnx_type))
        check_status(status)
        
        if onnx_type == LibOnnxRuntime::OnnxType::Tensor.value
          tensor_info = Pointer(Void).null
          status = api.cast_type_info_to_tensor_info.call(type_info, pointerof(tensor_info))
          check_status(status)
          
          element_type = 0
          status = api.get_tensor_element_type.call(tensor_info, pointerof(element_type))
          check_status(status)
          
          dims_count = 0_u64
          status = api.get_dimensions_count.call(tensor_info, pointerof(dims_count))
          check_status(status)
          
          dims = Array(Int64).new(dims_count, 0_i64)
          status = api.get_dimensions.call(tensor_info, dims.to_unsafe, dims_count)
          check_status(status)
          
          inputs << {name: name, type: LibOnnxRuntime::TensorElementDataType.new(element_type), shape: dims}
        end
        
        api.release_type_info.call(type_info)
      end
      
      inputs
    end

    private def load_outputs
      count = 0_u64
      status = api.session_get_output_count.call(@session, pointerof(count))
      check_status(status)

      outputs = [] of NamedTuple(name: String, type: LibOnnxRuntime::TensorElementDataType, shape: Array(Int64))
      
      count.times do |i|
        name_ptr = Pointer(Void).null
        name_len = 0_u64
        status = api.session_get_output_name.call(@session, i, name_ptr, pointerof(name_len))
        check_status(status)
        
        name_buffer = Bytes.new(name_len)
        status = api.session_get_output_name.call(@session, i, name_buffer.to_unsafe, pointerof(name_len))
        check_status(status)
        name = String.new(name_buffer)
        
        type_info = Pointer(Void).null
        status = api.session_get_output_type_info.call(@session, i, pointerof(type_info))
        check_status(status)
        
        onnx_type = 0
        status = api.get_onnx_type_from_type_info.call(type_info, pointerof(onnx_type))
        check_status(status)
        
        if onnx_type == LibOnnxRuntime::OnnxType::Tensor.value
          tensor_info = Pointer(Void).null
          status = api.cast_type_info_to_tensor_info.call(type_info, pointerof(tensor_info))
          check_status(status)
          
          element_type = 0
          status = api.get_tensor_element_type.call(tensor_info, pointerof(element_type))
          check_status(status)
          
          dims_count = 0_u64
          status = api.get_dimensions_count.call(tensor_info, pointerof(dims_count))
          check_status(status)
          
          dims = Array(Int64).new(dims_count, 0_i64)
          status = api.get_dimensions.call(tensor_info, dims.to_unsafe, dims_count)
          check_status(status)
          
          outputs << {name: name, type: LibOnnxRuntime::TensorElementDataType.new(element_type), shape: dims}
        end
        
        api.release_type_info.call(type_info)
      end
      
      outputs
    end

    def run(input_feed, output_names = nil, **run_options)
      run_options_ptr = api.create_run_options.call
      
      # Set run options if provided
      if run_options[:tag]?
        status = api.run_options_set_run_tag.call(run_options_ptr, run_options[:tag].to_s)
        check_status(status)
      end

      if run_options[:log_severity_level]?
        status = api.run_options_set_run_log_severity_level.call(run_options_ptr, run_options[:log_severity_level].to_i)
        check_status(status)
      end

      if run_options[:log_verbosity_level]?
        status = api.run_options_set_run_log_verbosity_level.call(run_options_ptr, run_options[:log_verbosity_level].to_i)
        check_status(status)
      end

      # Prepare input tensors
      input_tensors = [] of Pointer(Void)
      input_names = [] of String

      input_feed.each do |name, data|
        tensor = create_tensor(data)
        input_tensors << tensor
        input_names << name
      end

      # Prepare output names
      output_names = output_names || @outputs.map { |o| o[:name] }
      
      # Prepare output tensors
      output_tensors = Array(Pointer(Void)).new(output_names.size, Pointer(Void).null)

      # Run inference
      input_names_ptr = input_names.map { |name| ort_string(name) }
      output_names_ptr = output_names.map { |name| ort_string(name) }

      status = api.run.call(
        @session,
        run_options_ptr,
        input_names_ptr.to_unsafe,
        input_tensors.to_unsafe,
        input_tensors.size,
        output_names_ptr.to_unsafe,
        output_names.size,
        output_tensors.to_unsafe
      )
      check_status(status)

      # Clean up
      api.release_run_options.call(run_options_ptr)

      # Extract output data
      result = {} of String => Array(Float32) | Array(Int32) | Array(Int64)
      output_names.each_with_index do |name, i|
        tensor = output_tensors[i]
        
        type_info = Pointer(Void).null
        status = api.get_type_info.call(tensor, pointerof(type_info))
        check_status(status)
        
        onnx_type = 0
        status = api.get_onnx_type_from_type_info.call(type_info, pointerof(onnx_type))
        check_status(status)
        
        if onnx_type == LibOnnxRuntime::OnnxType::Tensor.value
          tensor_info = Pointer(Void).null
          status = api.cast_type_info_to_tensor_info.call(type_info, pointerof(tensor_info))
          check_status(status)
          
          element_type = 0
          status = api.get_tensor_element_type.call(tensor_info, pointerof(element_type))
          check_status(status)
          
          dims_count = 0_u64
          status = api.get_dimensions_count.call(tensor_info, pointerof(dims_count))
          check_status(status)
          
          dims = Array(Int64).new(dims_count, 0_i64)
          status = api.get_dimensions.call(tensor_info, dims.to_unsafe, dims_count)
          check_status(status)
          
          element_count = 1_i64
          dims.each { |d| element_count *= d }
          
          data_ptr = Pointer(Void).null
          status = api.get_tensor_mutable_data.call(tensor, pointerof(data_ptr))
          check_status(status)
          
          case LibOnnxRuntime::TensorElementDataType.new(element_type)
          when LibOnnxRuntime::TensorElementDataType::Float
            data = Slice.new(data_ptr.as(Float32*), element_count)
            result[name] = data.to_a
          when LibOnnxRuntime::TensorElementDataType::Int32
            data = Slice.new(data_ptr.as(Int32*), element_count)
            result[name] = data.to_a
          when LibOnnxRuntime::TensorElementDataType::Int64
            data = Slice.new(data_ptr.as(Int64*), element_count)
            result[name] = data.to_a
          else
            raise "Unsupported tensor element type: #{element_type}"
          end
        end
        
        api.release_type_info.call(type_info)
        api.release_value.call(tensor)
      end

      result
    end

    private def create_tensor(data)
      case data
      when Array(Float32)
        create_float_tensor(data)
      when Array(Int32)
        create_int32_tensor(data)
      when Array(Int64)
        create_int64_tensor(data)
      else
        raise "Unsupported data type: #{data.class}"
      end
    end

    private def create_float_tensor(data, shape = nil)
      shape = [data.size.to_i64] if shape.nil?
      tensor = Pointer(Void).null
      
      memory_info = api.create_cpu_memory_info.call(0, 0)
      
      status = api.create_tensor_with_data_as_ort_value.call(
        memory_info,
        data.to_unsafe.as(Void*),
        data.size * sizeof(Float32),
        shape.to_unsafe,
        shape.size,
        LibOnnxRuntime::TensorElementDataType::Float.value,
        pointerof(tensor)
      )
      check_status(status)
      
      api.release_memory_info.call(memory_info)
      tensor
    end

    private def create_int32_tensor(data, shape = nil)
      shape = [data.size.to_i64] if shape.nil?
      tensor = Pointer(Void).null
      
      memory_info = api.create_cpu_memory_info.call(0, 0)
      
      status = api.create_tensor_with_data_as_ort_value.call(
        memory_info,
        data.to_unsafe.as(Void*),
        data.size * sizeof(Int32),
        shape.to_unsafe,
        shape.size,
        LibOnnxRuntime::TensorElementDataType::Int32.value,
        pointerof(tensor)
      )
      check_status(status)
      
      api.release_memory_info.call(memory_info)
      tensor
    end

    private def create_int64_tensor(data, shape = nil)
      shape = [data.size.to_i64] if shape.nil?
      tensor = Pointer(Void).null
      
      memory_info = api.create_cpu_memory_info.call(0, 0)
      
      status = api.create_tensor_with_data_as_ort_value.call(
        memory_info,
        data.to_unsafe.as(Void*),
        data.size * sizeof(Int64),
        shape.to_unsafe,
        shape.size,
        LibOnnxRuntime::TensorElementDataType::Int64.value,
        pointerof(tensor)
      )
      check_status(status)
      
      api.release_memory_info.call(memory_info)
      tensor
    end

    private def check_status(status)
      return if status.null?
      
      error_code = api.get_error_code.call(status)
      error_message = String.new(api.get_error_message.call(status))
      api.release_status.call(status)
      
      raise "ONNXRuntime Error: #{error_message} (#{error_code})"
    end

    private def ort_string(str)
      str.to_unsafe
    end

    private def env
      @@env ||= begin
        env = Pointer(Void).null
        status = api.create_env.call(3, "onnxruntime.cr", pointerof(env))
        check_status(status)
        env
      end
    end
  end
end
