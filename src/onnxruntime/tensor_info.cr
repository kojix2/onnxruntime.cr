module OnnxRuntime
  # Represents metadata for a tensor (name, type, shape).
  class TensorInfo
    getter name : String
    getter type : LibOnnxRuntime::TensorElementDataType
    getter shape : Array(Int64)

    def initialize(@name, @type, @shape)
    end

    # Create a TensorInfo from OrtTypeInfo
    def self.from_type_info(name : String, type_info : Pointer(LibOnnxRuntime::OrtTypeInfo), session)
      api = session.api

      # Get ONNX type
      onnx_type = LibOnnxRuntime::OnnxType::TENSOR
      status = api.get_onnx_type_from_type_info.call(type_info, pointerof(onnx_type))
      session.check_status(status)

      case onnx_type
      when LibOnnxRuntime::OnnxType::TENSOR
        # Cast to tensor info
        tensor_info = Pointer(LibOnnxRuntime::OrtTensorTypeAndShapeInfo).null
        status = api.cast_type_info_to_tensor_info.call(type_info, pointerof(tensor_info))
        session.check_status(status)

        # Get element type
        element_type = LibOnnxRuntime::TensorElementDataType::FLOAT
        status = api.get_tensor_element_type.call(tensor_info, pointerof(element_type))
        session.check_status(status)

        # Get dimensions count
        dims_count = 0_u64
        status = api.get_dimensions_count.call(tensor_info, pointerof(dims_count))
        session.check_status(status)

        # Get dimensions
        dims = Array(Int64).new(dims_count, 0_i64)
        status = api.get_dimensions.call(tensor_info, dims.to_unsafe, dims_count)
        session.check_status(status)

        new(name, LibOnnxRuntime::TensorElementDataType.new(element_type), dims)
      else
        raise "Unsupported ONNX type: #{onnx_type}"
      end
    end
  end
end
