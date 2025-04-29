module OnnxRuntime
  # Abstract base class for execution providers
  abstract class Provider
    # Get the provider name
    abstract def name : String

    # Get the provider options
    abstract def options : ProviderOptions?
  end

  # Abstract base class for provider options
  abstract class ProviderOptions
    # Convert to C API representation
    abstract def to_unsafe
  end

  # CPU execution provider
  class CpuProvider < Provider
    def initialize(@options : CpuProviderOptions? = nil)
    end

    def name : String
      "CPUExecutionProvider"
    end

    def options : ProviderOptions?
      @options
    end
  end

  # CPU provider options
  class CpuProviderOptions < ProviderOptions
    # CPU provider options are not directly supported in the C API
    # We'll use a simple hash to store options
    @options = {} of String => String

    def initialize(use_arena = true)
      @options["use_arena"] = use_arena.to_s
    end

    # Get the underlying options as a hash
    def to_unsafe
      @options
    end
  end

  # CUDA execution provider
  class CudaProvider < Provider
    def initialize(@options : CudaProviderOptions? = nil)
    end

    def name : String
      "CUDAExecutionProvider"
    end

    def options : ProviderOptions?
      @options
    end
  end

  # CUDA provider options
  class CudaProviderOptions < ProviderOptions
    @options : Pointer(LibOnnxRuntime::OrtCUDAProviderOptionsV2)
    @released = false

    def initialize(session : InferenceSession, device_id = 0, gpu_mem_limit = 0_u64, arena_extend_strategy = 0, cudnn_conv_algo_search = 0, do_copy_in_default_stream = true)
      @api = session.api
      @options = create_cuda_provider_options

      # Set options as key-value pairs
      keys = ["device_id", "gpu_mem_limit", "arena_extend_strategy", "cudnn_conv_algo_search", "do_copy_in_default_stream"]
      values = [device_id.to_s, gpu_mem_limit.to_s, arena_extend_strategy.to_s, cudnn_conv_algo_search.to_s, do_copy_in_default_stream ? "1" : "0"]

      update_options(keys, values)
    end

    # Finalizer to release resources
    def finalize
      release
    end

    # Explicitly release resources
    def release
      return if @released
      @api.release_cuda_provider_options.call(@options) if @options
      @released = true
    end

    # Get the underlying OrtCUDAProviderOptionsV2 pointer
    def to_unsafe
      @options
    end

    private def create_cuda_provider_options
      options_ptr = Pointer(LibOnnxRuntime::OrtCUDAProviderOptionsV2).null
      status = @api.create_cuda_provider_options.call(pointerof(options_ptr))
      check_status(status)
      options_ptr
    end

    private def update_options(keys, values)
      keys_ptr = keys.map(&.to_unsafe)
      values_ptr = values.map(&.to_unsafe)

      status = @api.update_cuda_provider_options.call(
        @options,
        keys_ptr.to_unsafe,
        values_ptr.to_unsafe,
        keys.size
      )
      check_status(status)
    end

    private def check_status(status)
      return if status.null?

      error_code = @api.get_error_code.call(status)
      error_message = String.new(@api.get_error_message.call(status))
      @api.release_status.call(status)

      raise "ONNXRuntime Error: #{error_message} (#{error_code})"
    end
  end

  # Factory method to create providers
  def self.create_provider(name : String, session : InferenceSession, **options)
    case name.downcase
    when "cpu", "cpuexecutionprovider"
      use_arena = options["use_arena"]? || true
      CpuProvider.new(CpuProviderOptions.new(use_arena))
    when "cuda", "cudaexecutionprovider"
      device_id = options["device_id"]? || 0
      gpu_mem_limit = options["gpu_mem_limit"]? || 0_u64
      arena_extend_strategy = options["arena_extend_strategy"]? || 0
      cudnn_conv_algo_search = options["cudnn_conv_algo_search"]? || 0
      do_copy_in_default_stream = options["do_copy_in_default_stream"]? || true
      CudaProvider.new(CudaProviderOptions.new(
        session,
        device_id,
        gpu_mem_limit,
        arena_extend_strategy,
        cudnn_conv_algo_search,
        do_copy_in_default_stream
      ))
    else
      raise "Unsupported provider: #{name}"
    end
  end
end
