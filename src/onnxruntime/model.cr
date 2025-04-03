module OnnxRuntime
  class Model
    def initialize(path_or_bytes, **session_options)
      @session = InferenceSession.new(path_or_bytes, **session_options)
    end

    def predict(input_feed, output_names = nil, **run_options)
      # Convert input data to the appropriate format
      formatted_input = format_input(input_feed)

      # Run inference
      result = @session.run(formatted_input, output_names, **run_options)

      # Format output data if needed
      format_output(result)
    end

    # Create a COO format sparse tensor
    def create_coo_sparse_tensor(values, indices, dense_shape)
      SparseTensor.coo(values, indices, dense_shape)
    end

    # Create a CSR format sparse tensor
    def create_csr_sparse_tensor(values, inner_indices, outer_indices, dense_shape)
      SparseTensor.csr(values, inner_indices, outer_indices, dense_shape)
    end

    # Create a BlockSparse format sparse tensor
    def create_block_sparse_tensor(values, indices, dense_shape)
      SparseTensor.block_sparse(values, indices, dense_shape)
    end

    def inputs
      @session.inputs
    end

    def outputs
      @session.outputs
    end

    def metadata
      @session.modelmeta
    end

    private def format_input(input_feed)
      formatted = {} of String => Array(Float32) | Array(Int32) | Array(Int64) | SparseTensor

      input_feed.each do |name, data|
        # If data is already a SparseTensor, use it directly
        if data.is_a?(SparseTensor)
          formatted[name] = data
          next
        end

        # Find the input specification
        input_spec = @session.inputs.find { |i| i[:name] == name }
        raise "Unknown input: #{name}" unless input_spec

        # Convert data to the appropriate type and shape
        case input_spec[:type]
        when LibOnnxRuntime::TensorElementDataType::Float
          formatted[name] = convert_to_float32_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::Int32
          formatted[name] = convert_to_int32_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::Int64
          formatted[name] = convert_to_int64_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::Double
          formatted[name] = convert_to_float64_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::Uint8
          formatted[name] = convert_to_uint8_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::Int8
          formatted[name] = convert_to_int8_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::Uint16
          formatted[name] = convert_to_uint16_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::Int16
          formatted[name] = convert_to_int16_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::Uint32
          formatted[name] = convert_to_uint32_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::Uint64
          formatted[name] = convert_to_uint64_array(data, input_spec[:shape])
        when LibOnnxRuntime::TensorElementDataType::Bool
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
      when Array(Int)
        data.map(&.to_f32)
      else
        raise "Cannot convert #{data.class} to Array(Float32)"
      end
    end

    private def convert_to_float64_array(data, shape)
      case data
      when Array(Float64)
        data
      when Array(Float32)
        data.map(&.to_f64)
      when Array(Int)
        data.map(&.to_f64)
      else
        raise "Cannot convert #{data.class} to Array(Float64)"
      end
    end

    private def convert_to_int32_array(data, shape)
      case data
      when Array(Int32)
        data
      when Array(Int)
        data.map(&.to_i32)
      when Array(Float)
        data.map(&.to_i32)
      else
        raise "Cannot convert #{data.class} to Array(Int32)"
      end
    end

    private def convert_to_int64_array(data, shape)
      case data
      when Array(Int64)
        data
      when Array(Int)
        data.map(&.to_i64)
      when Array(Float)
        data.map(&.to_i64)
      else
        raise "Cannot convert #{data.class} to Array(Int64)"
      end
    end

    private def convert_to_uint8_array(data, shape)
      case data
      when Array(UInt8)
        data
      when Array(Int)
        data.map(&.to_u8)
      else
        raise "Cannot convert #{data.class} to Array(UInt8)"
      end
    end

    private def convert_to_int8_array(data, shape)
      case data
      when Array(Int8)
        data
      when Array(Int)
        data.map(&.to_i8)
      else
        raise "Cannot convert #{data.class} to Array(Int8)"
      end
    end

    private def convert_to_uint16_array(data, shape)
      case data
      when Array(UInt16)
        data
      when Array(Int)
        data.map(&.to_u16)
      else
        raise "Cannot convert #{data.class} to Array(UInt16)"
      end
    end

    private def convert_to_int16_array(data, shape)
      case data
      when Array(Int16)
        data
      when Array(Int)
        data.map(&.to_i16)
      else
        raise "Cannot convert #{data.class} to Array(Int16)"
      end
    end

    private def convert_to_uint32_array(data, shape)
      case data
      when Array(UInt32)
        data
      when Array(Int)
        data.map(&.to_u32)
      else
        raise "Cannot convert #{data.class} to Array(UInt32)"
      end
    end

    private def convert_to_uint64_array(data, shape)
      case data
      when Array(UInt64)
        data
      when Array(Int)
        data.map(&.to_u64)
      else
        raise "Cannot convert #{data.class} to Array(UInt64)"
      end
    end

    private def convert_to_bool_array(data, shape)
      case data
      when Array(Bool)
        data
      when Array(Int)
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
