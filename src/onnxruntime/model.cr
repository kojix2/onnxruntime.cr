module OnnxRuntime
  class Model
    def initialize(path_or_bytes, **session_options)
      @session = InferenceSession.new(path_or_bytes, **session_options)
    end

    # Finalizer to release session resources
    def finalize
      release
    end

    # Explicitly release resources
    def release
      @session.release_session if @session
    end

    def predict(input_feed, output_names = nil, shape = nil, **run_options)
      # Convert input data to the appropriate format
      formatted_input = format_input(input_feed)

      # Run inference with custom shapes if provided
      if shape
        # Create tensors with custom shapes
        input_tensors = NamedTensors.new
        formatted_input.each do |name, data|
          if shape[name]? && data.is_a?(Array)
            # Use the custom shape for this input
            input_tensors[name] = data
          else
            # Use the data as is
            input_tensors[name] = data
          end
        end

        # Add shape information to run options
        run_options = run_options.merge({shape: shape})

        # Run inference
        result = @session.run(input_tensors, output_names, **run_options)
      else
        # Run inference without custom shapes, using the original input data
        shape = @session.inputs.map { |i| {i[:name], i[:shape]} }.to_h.select(input_feed.keys)
        run_options = run_options.merge({shape: shape})
        result = @session.run(formatted_input, output_names, **run_options)
      end

      # Format output data if needed
      format_output(result)
    end

    # Create a COO format sparse tensor
    def create_coo_sparse_tensor(values : Array(T), indices, dense_shape) forall T
      SparseTensor(T).coo(values, indices, dense_shape)
    end

    # Create a CSR format sparse tensor
    def create_csr_sparse_tensor(values : Array(T), inner_indices, outer_indices, dense_shape) forall T
      SparseTensor(T).csr(values, inner_indices, outer_indices, dense_shape)
    end

    # Create a BlockSparse format sparse tensor
    def create_block_sparse_tensor(values : Array(T), indices, dense_shape) forall T
      SparseTensor(T).block_sparse(values, indices, dense_shape)
    end

    def inputs
      @session.inputs
    end

    def outputs
      @session.outputs
    end

    def metadata
      {
        "inputs"  => @session.inputs,
        "outputs" => @session.outputs,
      }
    end

    private def format_input(input_feed)
      formatted = NamedTensors.new

      input_feed.each do |name, data|
        # If data is already a SparseTensor, use it directly
        if data.is_a?(SparseTensorFloat32) || data.is_a?(SparseTensorInt32) || data.is_a?(SparseTensorInt64) || data.is_a?(SparseTensorFloat64)
          formatted[name] = data
          next
        end

        # Find the input specification
        input_spec = @session.inputs.find { |i| i[:name] == name }
        raise "Unknown input: #{name}" unless input_spec

        # Convert data to the appropriate type and shape
        case input_spec[:type]
        when LibOnnxRuntime::TensorElementDataType::FLOAT
          formatted[name] = convert_to_float32_array(data)
        when LibOnnxRuntime::TensorElementDataType::INT32
          formatted[name] = convert_to_int32_array(data)
        when LibOnnxRuntime::TensorElementDataType::INT64
          formatted[name] = convert_to_int64_array(data)
        when LibOnnxRuntime::TensorElementDataType::DOUBLE
          formatted[name] = convert_to_float64_array(data)
        when LibOnnxRuntime::TensorElementDataType::UINT8
          formatted[name] = convert_to_uint8_array(data)
        when LibOnnxRuntime::TensorElementDataType::INT8
          formatted[name] = convert_to_int8_array(data)
        when LibOnnxRuntime::TensorElementDataType::UINT16
          formatted[name] = convert_to_uint16_array(data)
        when LibOnnxRuntime::TensorElementDataType::INT16
          formatted[name] = convert_to_int16_array(data)
        when LibOnnxRuntime::TensorElementDataType::UINT32
          formatted[name] = convert_to_uint32_array(data)
        when LibOnnxRuntime::TensorElementDataType::UINT64
          formatted[name] = convert_to_uint64_array(data)
        when LibOnnxRuntime::TensorElementDataType::BOOL
          formatted[name] = convert_to_bool_array(data)
        else
          raise "Unsupported input type: #{input_spec[:type]}"
        end
      end

      formatted
    end

    private def convert_to_float32_array(data : Array(Float32))
      data
    end

    private def convert_to_float32_array(data : Array(Float64) | Array(Int32) | Array(Int64))
      data.map(&.to_f32)
    end

    private def convert_to_float32_array(data)
      raise "Cannot convert #{data.class} to Array(Float32)"
    end

    private def convert_to_float64_array(data : Array(Float64))
      data
    end

    private def convert_to_float64_array(data : Array(Float32) | Array(Int32) | Array(Int64))
      data.map(&.to_f64)
    end

    private def convert_to_float64_array(data)
      raise "Cannot convert #{data.class} to Array(Float64)"
    end

    private def convert_to_int32_array(data : Array(Int32))
      data
    end

    private def convert_to_int32_array(data : Array(Int64) | Array(Float32) | Array(Float64))
      data.map(&.to_i32)
    end

    private def convert_to_int32_array(data)
      raise "Cannot convert #{data.class} to Array(Int32)"
    end

    private def convert_to_int64_array(data : Array(Int64))
      data
    end

    private def convert_to_int64_array(data : Array(Int32) | Array(Float32) | Array(Float64))
      data.map(&.to_i64)
    end

    private def convert_to_int64_array(data)
      raise "Cannot convert #{data.class} to Array(Int64)"
    end

    private def convert_to_uint8_array(data : Array(UInt8))
      data
    end

    private def convert_to_uint8_array(data : Array(Int32) | Array(Int64))
      data.map(&.to_u8)
    end

    private def convert_to_uint8_array(data)
      raise "Cannot convert #{data.class} to Array(UInt8)"
    end

    private def convert_to_int8_array(data : Array(Int8))
      data
    end

    private def convert_to_int8_array(data : Array(Int32) | Array(Int64))
      data.map(&.to_i8)
    end

    private def convert_to_int8_array(data)
      raise "Cannot convert #{data.class} to Array(Int8)"
    end

    private def convert_to_uint16_array(data : Array(UInt16))
      data
    end

    private def convert_to_uint16_array(data : Array(Int32) | Array(Int64))
      data.map(&.to_u16)
    end

    private def convert_to_uint16_array(data)
      raise "Cannot convert #{data.class} to Array(UInt16)"
    end

    private def convert_to_int16_array(data : Array(Int16))
      data
    end

    private def convert_to_int16_array(data : Array(Int32) | Array(Int64))
      data.map(&.to_i16)
    end

    private def convert_to_int16_array(data)
      raise "Cannot convert #{data.class} to Array(Int16)"
    end

    private def convert_to_uint32_array(data : Array(UInt32))
      data
    end

    private def convert_to_uint32_array(data : Array(Int32) | Array(Int64))
      data.map(&.to_u32)
    end

    private def convert_to_uint32_array(data)
      raise "Cannot convert #{data.class} to Array(UInt32)"
    end

    private def convert_to_uint64_array(data : Array(UInt64))
      data
    end

    private def convert_to_uint64_array(data : Array(Int32) | Array(Int64))
      data.map(&.to_u64)
    end

    private def convert_to_uint64_array(data)
      raise "Cannot convert #{data.class} to Array(UInt64)"
    end

    private def convert_to_bool_array(data : Array(Bool))
      data
    end

    private def convert_to_bool_array(data : Array(Int32) | Array(Int64))
      data.map { |v| v != 0 }
    end

    private def convert_to_bool_array(data)
      raise "Cannot convert #{data.class} to Array(Bool)"
    end

    private def format_output(result)
      # Return the result as is, including any SparseTensor objects
      result
    end
  end
end
