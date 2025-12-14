module OnnxRuntime
  class OrtEnvironment
    @@instance : OrtEnvironment?
    @@mutex = Mutex.new

    @env : Pointer(LibOnnxRuntime::OrtEnv)
    @released = false
    @release_in_progress = false

    def self.instance
      @@mutex.synchronize do
        @@instance ||= new
      end
    end

    private def initialize
      @env = create_env
      @released = false
      @release_in_progress = false
    end

    def env
      raise "Environment has been released" if @released
      @env
    end

    def released?
      @released
    end

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

    def api
      OnnxRuntime::LibOnnxRuntime
        .OrtGetApiBase.value
        .get_api
        .call(OnnxRuntime::LibOnnxRuntime::ORT_API_VERSION)
        .value
    end

    private def create_env
      env = Pointer(LibOnnxRuntime::OrtEnv).null
      status = api.create_env.call(
        OnnxRuntime::LibOnnxRuntime::LoggingLevel::ERROR,
        ort_env_string("onnxruntime.cr"),
        pointerof(env)
      )
      check_status(status)
      env
    end

    private def check_status(status)
      return if status.null?

      error_code = api.get_error_code.call(status)
      error_message = String.new(api.get_error_message.call(status))
      api.release_status.call(status)

      raise "ONNXRuntime Error: #{error_message} (#{error_code})"
    end

    private def ort_env_string(str : String)
      {% if flag?(:win32) %}
        utf16 = str.to_utf16
        utf16.to_unsafe.as(OnnxRuntime::LibOnnxRuntime::ORTCHAR_T*)
      {% else %}
        str.to_unsafe
      {% end %}
    end
  end
end
