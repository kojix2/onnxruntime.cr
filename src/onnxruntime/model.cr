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
      formatted = {} of String => Array(Float32) | Array(Int32) | Array(Int64)
      
      input_feed.each do |name, data|
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

    private def format_output(result)
      # For now, just return the raw result
      # This could be extended to reshape arrays according to output shapes
      result
    end
  end
end
