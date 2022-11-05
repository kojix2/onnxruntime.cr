module OnnxRuntime
  class InferenceSession
    getter :inputs, :outputs
    def initialize(path_or_bytes)
      session_options = nil
      @session = load_session(path_or_bytes, session_options)
      @allocator = load_allocator
      @inputs = load_inputs
      @outputs = load_outputs
    end

    private def load_allocator
    end

    private def load_inputs
    end

    private def load_outputs
    end

    private def check_status(status)
    end
  end
end