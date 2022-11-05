module OnnxRuntime
  class InferenceSession
    getter :inputs, :outputs

    def api
      OnnxRuntime::LibOnnxRuntime
        .OrtGetApiBase.value
        .get_api
        .call(OnnxRuntime::LibOnnxRuntime::ORT_API_VERSION)
        .value
    end

    def initialize(path_or_bytes)
      session_options = Pointer(Void).null
      api.create_session_options.call(session_options)
      @session = load_session(path_or_bytes, session_options)
      @allocator = load_allocator
      @inputs = load_inputs
      @outputs = load_outputs
    end

    private def load_session(path_or_bytes, session_options)
      session = Pointer(Void).null
      # api.create_session.call(env.read_pointer, ort_string(path_or_bytes), session_options.read_pointer, session)
      return 0
    end

    private def load_allocator
    end

    private def load_inputs
    end

    private def load_outputs
    end

    private def check_status(status)
    end

    private def ort_string
    end

    private def env
    end
  end
end
