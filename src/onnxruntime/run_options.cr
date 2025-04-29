module OnnxRuntime
  # RunOptions class provides options for inference execution.
  class RunOptions
    @run_options : Pointer(LibOnnxRuntime::OrtRunOptions)
    @released = false

    # Creates a new RunOptions instance.
    def initialize(session : InferenceSession)
      @api = session.api
      @run_options = create_run_options
    end

    # Finalizer to release resources
    def finalize
      release
    end

    # Explicitly release resources
    def release
      return if @released
      @api.release_run_options.call(@run_options) if @run_options
      @released = true
    end

    # Set run tag
    def tag=(tag : String)
      status = @api.run_options_set_run_tag.call(@run_options, tag)
      check_status(status)
      self
    end

    # Set run log severity level
    def log_severity_level=(level : Int32)
      status = @api.run_options_set_run_log_severity_level.call(@run_options, level)
      check_status(status)
      self
    end

    # Set run log verbosity level
    def log_verbosity_level=(level : Int32)
      status = @api.run_options_set_run_log_verbosity_level.call(@run_options, level)
      check_status(status)
      self
    end

    # Set terminate flag
    def terminate
      status = @api.run_options_set_terminate.call(@run_options)
      check_status(status)
      self
    end

    # Get the underlying OrtRunOptions pointer
    def to_unsafe
      @run_options
    end

    private def create_run_options
      run_options_ptr = Pointer(LibOnnxRuntime::OrtRunOptions).null
      status = @api.create_run_options.call(pointerof(run_options_ptr))
      check_status(status)
      run_options_ptr
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
