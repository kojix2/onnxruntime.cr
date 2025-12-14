module OnnxRuntime
  # IoBinding class provides zero-copy inference and advanced input/output binding.
  # This is a placeholder for future implementation.
  class IoBinding
    @io_binding : Pointer(LibOnnxRuntime::OrtIoBinding)
    @released = false

    # Creates a new IoBinding instance.
    def initialize(session : InferenceSession)
      @api = session.api
      @session = session
      @io_binding = create_io_binding
    end

    # Finalizer to release resources
    def finalize
      release
    end

    # Explicitly release resources
    def release
      return if @released
      @api.release_io_binding.call(@io_binding) if @io_binding
      @released = true
    end

    # Bind input to a tensor
    def bind_input(name : String, tensor_ptr : Pointer(LibOnnxRuntime::OrtValue))
      status = @api.bind_input.call(@io_binding, name, tensor_ptr)
      check_status(status)
      self
    end

    # Bind output to a tensor
    def bind_output(name : String, tensor_ptr : Pointer(LibOnnxRuntime::OrtValue))
      status = @api.bind_output.call(@io_binding, name, tensor_ptr)
      check_status(status)
      self
    end

    # Get the underlying OrtIoBinding pointer
    def to_unsafe
      @io_binding
    end

    private def create_io_binding
      io_binding_ptr = Pointer(LibOnnxRuntime::OrtIoBinding).null
      status = @api.create_io_binding.call(@session.session, pointerof(io_binding_ptr))
      check_status(status)
      io_binding_ptr
    end

    private def check_status(status)
      return if status.null?

      error_code = @api.get_error_code.call(status)
      error_message = String.new(@api.get_error_message.call(status))
      @api.release_status.call(status)

      raise "ONNXRuntime Error: #{error_message} (#{error_code})"
    end
  end
end
