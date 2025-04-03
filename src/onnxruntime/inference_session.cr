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
      session_options_ptr = Pointer(Void).null
      status = api.create_session_options.call(pointerof(session_options_ptr))
      check_status(status)

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
      run_options_ptr = Pointer(Void).null
      status = api.create_run_options.call(pointerof(run_options_ptr))
      check_status(status)

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
        tensor = if data.is_a?(SparseTensor)
                   data.to_ort_value(self)
                 else
                   create_tensor(data)
                 end
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
      result = {} of String => Array(Float32) | Array(Int32) | Array(Int64) | SparseTensor
      output_names.each_with_index do |name, i|
        tensor = output_tensors[i]

        type_info = Pointer(Void).null
        status = api.get_type_info.call(tensor, pointerof(type_info))
        check_status(status)

        onnx_type = 0
        status = api.get_onnx_type_from_type_info.call(type_info, pointerof(onnx_type))
        check_status(status)

        if onnx_type == LibOnnxRuntime::OnnxType::Tensor.value
          # Handle dense tensor
          result[name] = extract_dense_tensor_data(tensor, type_info)
        elsif onnx_type == LibOnnxRuntime::OnnxType::SparseTensor.value
          # Handle sparse tensor
          result[name] = extract_sparse_tensor_data(tensor)
        else
          raise "Unsupported ONNX type: #{onnx_type}"
        end

        api.release_type_info.call(type_info)
        api.release_value.call(tensor)
      end

      result
    end

    # Extract data from a dense tensor
    private def extract_dense_tensor_data(tensor, type_info)
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
        data.to_a
      when LibOnnxRuntime::TensorElementDataType::Int32
        data = Slice.new(data_ptr.as(Int32*), element_count)
        data.to_a
      when LibOnnxRuntime::TensorElementDataType::Int64
        data = Slice.new(data_ptr.as(Int64*), element_count)
        data.to_a
      when LibOnnxRuntime::TensorElementDataType::Double
        data = Slice.new(data_ptr.as(Float64*), element_count)
        data.to_a.map(&.to_f32)
      when LibOnnxRuntime::TensorElementDataType::Uint8
        data = Slice.new(data_ptr.as(UInt8*), element_count)
        data.to_a
      when LibOnnxRuntime::TensorElementDataType::Int8
        data = Slice.new(data_ptr.as(Int8*), element_count)
        data.to_a
      when LibOnnxRuntime::TensorElementDataType::Uint16
        data = Slice.new(data_ptr.as(UInt16*), element_count)
        data.to_a
      when LibOnnxRuntime::TensorElementDataType::Int16
        data = Slice.new(data_ptr.as(Int16*), element_count)
        data.to_a
      when LibOnnxRuntime::TensorElementDataType::Uint32
        data = Slice.new(data_ptr.as(UInt32*), element_count)
        data.to_a
      when LibOnnxRuntime::TensorElementDataType::Uint64
        data = Slice.new(data_ptr.as(UInt64*), element_count)
        data.to_a
      when LibOnnxRuntime::TensorElementDataType::Bool
        data = Slice.new(data_ptr.as(Bool*), element_count)
        data.to_a
      else
        raise "Unsupported tensor element type: #{element_type}"
      end
    end

    # Extract data from a sparse tensor
    private def extract_sparse_tensor_data(tensor)
      # Get sparse tensor format
      format = LibOnnxRuntime::SparseFormat::UNDEFINED
      status = api.get_sparse_tensor_format.call(tensor, pointerof(format))
      check_status(status)

      # Get values type and shape
      values_info = Pointer(Void).null
      status = api.get_sparse_tensor_values_type_and_shape.call(tensor, pointerof(values_info))
      check_status(status)

      # Get element type
      element_type = 0
      status = api.get_tensor_element_type.call(values_info, pointerof(element_type))
      check_status(status)

      # Get values shape
      dims_count = 0_u64
      status = api.get_dimensions_count.call(values_info, pointerof(dims_count))
      check_status(status)

      values_shape = Array(Int64).new(dims_count, 0_i64)
      status = api.get_dimensions.call(values_info, values_shape.to_unsafe, dims_count)
      check_status(status)

      # Get values data
      values_tensor = Pointer(Void).null
      status = api.get_sparse_tensor_values.call(tensor, pointerof(values_tensor))
      check_status(status)

      data_ptr = Pointer(Void).null
      status = api.get_tensor_mutable_data.call(values_tensor, pointerof(data_ptr))
      check_status(status)

      # Calculate total number of values
      values_count = 1_i64
      values_shape.each { |d| values_count *= d }

      # Extract values based on element type
      values = case LibOnnxRuntime::TensorElementDataType.new(element_type)
               when LibOnnxRuntime::TensorElementDataType::Float
                 Slice.new(data_ptr.as(Float32*), values_count).to_a
               when LibOnnxRuntime::TensorElementDataType::Int32
                 Slice.new(data_ptr.as(Int32*), values_count).to_a
               when LibOnnxRuntime::TensorElementDataType::Int64
                 Slice.new(data_ptr.as(Int64*), values_count).to_a
               when LibOnnxRuntime::TensorElementDataType::Double
                 Slice.new(data_ptr.as(Float64*), values_count).to_a
               else
                 raise "Unsupported sparse tensor element type: #{element_type}"
               end

      # Extract indices based on format
      indices = {} of Symbol => Array

      case format
      when LibOnnxRuntime::SparseFormat::COO
        indices[:coo_indices] = extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::COO_INDICES)
      when LibOnnxRuntime::SparseFormat::CSRC
        indices[:inner_indices] = extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::CSR_INNER_INDICES)
        indices[:outer_indices] = extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::CSR_OUTER_INDICES)
      when LibOnnxRuntime::SparseFormat::BLOCK_SPARSE
        indices[:block_indices] = extract_sparse_indices(tensor, LibOnnxRuntime::SparseIndicesFormat::BLOCK_SPARSE_INDICES)
      else
        raise "Unsupported sparse format: #{format}"
      end

      # Get dense shape from the first output
      dense_shape = @outputs.first[:shape]

      # Create and return SparseTensor
      SparseTensor.new(format, values, indices, dense_shape)
    end

    # Extract indices from a sparse tensor
    private def extract_sparse_indices(tensor, indices_format)
      # Get indices type and shape
      indices_info = Pointer(Void).null
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
      indices_tensor = Pointer(Void).null
      status = api.get_sparse_tensor_indices.call(tensor, indices_format, pointerof(indices_tensor))
      check_status(status)

      data_ptr = Pointer(Void).null
      status = api.get_tensor_mutable_data.call(indices_tensor, pointerof(data_ptr))
      check_status(status)

      # Calculate total number of indices
      indices_count = 1_i64
      indices_shape.each { |d| indices_count *= d }

      # Extract indices
      if indices_format == LibOnnxRuntime::SparseIndicesFormat::BLOCK_SPARSE_INDICES
        Slice.new(data_ptr.as(Int32*), indices_count).to_a
      else
        Slice.new(data_ptr.as(Int64*), indices_count).to_a
      end
    end

    private def create_tensor(data)
      case data
      when Array(Float32)
        create_float_tensor(data)
      when Array(Int32)
        create_int32_tensor(data)
      when Array(Int64)
        create_int64_tensor(data)
      when Array(Float64)
        create_float64_tensor(data)
      when Array(UInt8)
        create_uint8_tensor(data)
      when Array(Int8)
        create_int8_tensor(data)
      when Array(UInt16)
        create_uint16_tensor(data)
      when Array(Int16)
        create_int16_tensor(data)
      when Array(UInt32)
        create_uint32_tensor(data)
      when Array(UInt64)
        create_uint64_tensor(data)
      when Array(Bool)
        create_bool_tensor(data)
      else
        raise "Unsupported data type: #{data.class}"
      end
    end

    private def create_float_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Float, sizeof(Float32))
    end

    private def create_int32_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Int32, sizeof(Int32))
    end

    private def create_int64_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Int64, sizeof(Int64))
    end

    private def create_float64_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Double, sizeof(Float64))
    end

    private def create_uint8_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Uint8, sizeof(UInt8))
    end

    private def create_int8_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Int8, sizeof(Int8))
    end

    private def create_uint16_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Uint16, sizeof(UInt16))
    end

    private def create_int16_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Int16, sizeof(Int16))
    end

    private def create_uint32_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Uint32, sizeof(UInt32))
    end

    private def create_uint64_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Uint64, sizeof(UInt64))
    end

    private def create_bool_tensor(data, shape = nil)
      create_tensor_with_data(data, shape, LibOnnxRuntime::TensorElementDataType::Bool, sizeof(Bool))
    end

    private def create_tensor_with_data(data, shape, element_type, element_size)
      shape = [data.size.to_i64] if shape.nil?
      tensor = Pointer(Void).null

      memory_info = Pointer(Void).null
      status = api.create_cpu_memory_info.call(0, 0, pointerof(memory_info))
      check_status(status)

      status = api.create_tensor_with_data_as_ort_value.call(
        memory_info,
        data.to_unsafe.as(Void*),
        data.size * element_size,
        shape.to_unsafe,
        shape.size,
        element_type.value,
        pointerof(tensor)
      )
      check_status(status)

      api.release_memory_info.call(memory_info)
      tensor
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
