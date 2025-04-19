module OnnxRuntime
  # OrtEnvironment is a singleton class that manages the ONNX Runtime environment.
  # It ensures that only one environment exists per process and provides thread-safe access.
  class OrtEnvironment
    @@instance : OrtEnvironment?
    @@mutex = Mutex.new

    @env : Pointer(LibOnnxRuntime::OrtEnv)
    @released = false
    @release_in_progress = false

    # Get the singleton instance of OrtEnvironment
    def self.instance
      @@mutex.synchronize do
        @@instance ||= new
      end
    end

    # Private constructor to enforce singleton pattern
    private def initialize
      @env = create_env
      @released = false
      @release_in_progress = false
    end

    # Get the environment pointer
    def env
      raise "Environment has been released" if @released
      @env
    end

    # Check if the environment has been released
    def released?
      @released
    end

    # Release the environment in a thread-safe manner
    def release
      return if @released

      @@mutex.synchronize do
        return if @released

        begin
          api.release_env.call(@env)
          @released = true
          @@instance = nil
        rescue ex
          # Log error but don't propagate it during shutdown
          STDERR.puts "Error releasing ONNX Runtime environment: #{ex.message}"
        end
      end
    end

    # Get the API
    def api
      OnnxRuntime::LibOnnxRuntime
        .OrtGetApiBase.value
        .get_api
        .call(OnnxRuntime::LibOnnxRuntime::ORT_API_VERSION)
        .value
    end

    # Create a new environment
    private def create_env
      env = Pointer(LibOnnxRuntime::OrtEnv).null
      status = api.create_env.call(
        OnnxRuntime::LibOnnxRuntime::LoggingLevel::ERROR,
        "onnxruntime.cr".to_unsafe,
        pointerof(env)
      )
      check_status(status)
      env
    end

    # Check the status and raise an error if needed
    private def check_status(status)
      return if status.null?

      error_code = api.get_error_code.call(status)
      error_message = String.new(api.get_error_message.call(status))
      api.release_status.call(status)

      raise "ONNXRuntime Error: #{error_message} (#{error_code})"
    end
  end
end
