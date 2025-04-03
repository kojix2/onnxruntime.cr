module OnnxRuntime
  class Model
    def initialize(path_or_bytes, **session_options)
      @session = InferenceSession.new(path_or_bytes, **session_options)
    end

    def predict(input_feed, output_names = nil, shape = nil, **run_options)
      # Convert input data to the appropriate format
      formatted_input = format_input(input_feed)

      # Run inference with custom shapes if provided
      if shape
        # Create tensors with custom shapes
        input_tensors = {} of String => Array(Float32) | Array(Int32) | Array(Int64) | Array(Bool) | Array(UInt8) | Array(Int8) | Array(UInt16) | Array(Int16) | Array(UInt32) | Array(UInt64) | SparseTensorFloat32 | SparseTensorInt32 | SparseTensorInt64 | SparseTensorFloat64

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
        # Run inference without custom shapes
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
      formatted = {} of String => Array(Float32) | Array(Int32) | Array(Int64) | Array(Bool) | Array(UInt8) | Array(Int8) | Array(UInt16) | Array(Int16) | Array(UInt32) | Array(UInt64) | SparseTensorFloat32 | SparseTensorInt32 | SparseTensorInt64 | SparseTensorFloat64

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
          formatted[name] = convert_to_float32_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::INT32
          formatted[name] = convert_to_int32_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::INT64
          formatted[name] = convert_to_int64_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::DOUBLE
          formatted[name] = convert_to_float64_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::UINT8
          formatted[name] = convert_to_uint8_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::INT8
          formatted[name] = convert_to_int8_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::UINT16
          formatted[name] = convert_to_uint16_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::INT16
          formatted[name] = convert_to_int16_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::UINT32
          formatted[name] = convert_to_uint32_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::UINT64
          formatted[name] = convert_to_uint64_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::BOOL
          formatted[name] = convert_to_bool_array(data, input_spec[:shape])
        else
          raise "Unsupported input type: #{input_spec[:type]}"
        end
      end

      formatted
    end

    private def convert_to_float32_array(data, shape)
      case data
      when Array(Float32)
        data
      when Array(Float64)
        data.map(&.to_f32)
      when Array(Int32)
        data.map(&.to_f32)
      when Array(Int64)
        data.map(&.to_f32)
      else
        raise "Cannot convert #{data.class} to Array(Float32)"
      end
    end

    private def convert_to_float64_array(data, shape)
      case data
      when Array(Float64)
        data.map(&.to_f32)
      when Array(Float32)
        data
      when Array(Int32)
        data.map(&.to_f32)
      when Array(Int64)
        data.map(&.to_f32)
      else
        raise "Cannot convert #{data.class} to Array(Float32)"
      end
    end

    private def convert_to_int32_array(data, shape)
      case data
      when Array(Int32)
        data
      when Array(Int64)
        data.map(&.to_i32)
      when Array(Float32)
        data.map(&.to_i32)
      when Array(Float64)
        data.map(&.to_i32)
      else
        raise "Cannot convert #{data.class} to Array(Int32)"
      end
    end

    private def convert_to_int64_array(data, shape)
      case data
      when Array(Int64)
        data
      when Array(Int32)
        data.map(&.to_i64)
      when Array(Float32)
        data.map(&.to_i64)
      when Array(Float64)
        data.map(&.to_i64)
      else
        raise "Cannot convert #{data.class} to Array(Int64)"
      end
    end

    private def convert_to_uint8_array(data, shape)
      case data
      when Array(UInt8)
        data
      when Array(Int32)
        data.map(&.to_u8)
      when Array(Int64)
        data.map(&.to_u8)
      else
        raise "Cannot convert #{data.class} to Array(UInt8)"
      end
    end

    private def convert_to_int8_array(data, shape)
      case data
      when Array(Int8)
        data
      when Array(Int32)
        data.map(&.to_i8)
      when Array(Int64)
        data.map(&.to_i8)
      else
        raise "Cannot convert #{data.class} to Array(Int8)"
      end
    end

    private def convert_to_uint16_array(data, shape)
      case data
      when Array(UInt16)
        data
      when Array(Int32)
        data.map(&.to_u16)
      when Array(Int64)
        data.map(&.to_u16)
      else
        raise "Cannot convert #{data.class} to Array(UInt16)"
      end
    end

    private def convert_to_int16_array(data, shape)
      case data
      when Array(Int16)
        data
      when Array(Int32)
        data.map(&.to_i16)
      when Array(Int64)
        data.map(&.to_i16)
      else
        raise "Cannot convert #{data.class} to Array(Int16)"
      end
    end

    private def convert_to_uint32_array(data, shape)
      case data
      when Array(UInt32)
        data
      when Array(Int32)
        data.map(&.to_u32)
      when Array(Int64)
        data.map(&.to_u32)
      else
        raise "Cannot convert #{data.class} to Array(UInt32)"
      end
    end

    private def convert_to_uint64_array(data, shape)
      case data
      when Array(UInt64)
        data
      when Array(Int32)
        data.map(&.to_u64)
      when Array(Int64)
        data.map(&.to_u64)
      else
        raise "Cannot convert #{data.class} to Array(UInt64)"
      end
    end

    private def convert_to_bool_array(data, shape)
      case data
      when Array(Bool)
        data
      when Array(Int32)
        data.map { |v| v != 0 }
      when Array(Int64)
        data.map { |v| v != 0 }
      else
        raise "Cannot convert #{data.class} to Array(Bool)"
      end
    end

    private def format_output(result)
      # Return the result as is, including any SparseTensor objects
      result
    end
  end
end
