module OnnxRuntime
  {% if env("ONNXRUNTIMEDIR") %}
    @[Link(ldflags: "-L `echo $ONNXRUNTIMEDIR/lib` -lonnxruntime -Wl,-rpath,`echo $ONNXRUNTIMEDIR/lib`")]
  {% else %}
    @[Link("onnxruntime")]
  {% end %}
  lib LibOnnxRuntime
    ORT_API_VERSION = 21.as(UInt32)

    # Copied from TensorProto::DataType
    enum TensorElementDataType
      UNDEFINED  = 0
      FLOAT  # maps to c type float
      UINT8  # maps to c type uint8_t
      INT8   # maps to c type int8_t
      UINT16 # maps to c type uint16_t
      INT16  # maps to c type int16_t
      INT32  # maps to c type int32_t
      INT64  # maps to c type int64_t
      STRING # maps to c++ type std::string
      BOOL
      FLOAT16
      DOUBLE     # maps to c type double
      UINT32     # maps to c type uint32_t
      UINT64     # maps to c type uint64_t
      COMPLEX64  # complex with float32 real and imaginary components
      COMPLEX128 # complex with float64 real and imaginary components
      BFLOAT16   # Non-IEEE floating-point format based on IEEE754 single-precision
      # float 8 types were introduced in onnx 1.14
      FLOAT8E4M3FN   # Non-IEEE floating-point format based on IEEE754 single-precision
      FLOAT8E4M3FNUZ # Non-IEEE floating-point format based on IEEE754 single-precision
      FLOAT8E5M2     # Non-IEEE floating-point format based on IEEE754 single-precision
      FLOAT8E5M2FNUZ # Non-IEEE floating-point format based on IEEE754 single-precision
      # Int4 types were introduced in ONNX 1.16
      UINT4 # maps to a pair of packed uint4 values (size == 1 byte)
      INT4  # maps to a pair of packed int4 values (size == 1 byte)
    end

    # For C API compatibility
    alias ONNXTensorElementDataType = TensorElementDataType

    # Synced with onnx TypeProto oneof
    enum OnnxType
      UNKNOWN      = 0
      TENSOR
      SEQUENCE
      MAP
      OPAQUE
      SPARSETENSOR
      OPTIONAL
    end

    # For C API compatibility
    alias ONNXType = OnnxType

    # These types are synced with internal SparseFormatFlags
    enum SparseFormat
      UNDEFINED    =   0
      COO          = 0x1
      CSRC         = 0x2
      BLOCK_SPARSE = 0x4
    end

    # For C API compatibility
    alias OrtSparseFormat = SparseFormat

    # Enum allows to query sparse tensor indices
    enum SparseIndicesFormat
      COO_INDICES          = 0
      CSR_INNER_INDICES
      CSR_OUTER_INDICES
      BLOCK_SPARSE_INDICES
    end

    # For C API compatibility
    alias OrtSparseIndicesFormat = SparseIndicesFormat

    # Logging severity levels
    enum LoggingLevel
      VERBOSE = 0 # Verbose informational messages (least severe)
      INFO        # Informational messages
      WARNING     # Warning messages
      ERROR       # Error messages
      FATAL       # Fatal error messages (most severe)
    end

    # For C API compatibility
    alias OrtLoggingLevel = LoggingLevel

    enum ErrorCode
      OK                = 0
      FAIL
      INVALID_ARGUMENT
      NO_SUCHFILE
      NO_MODEL
      ENGINE_ERROR
      RUNTIME_EXCEPTION
      INVALID_PROTOBUF
      MODEL_LOADED
      NOT_IMPLEMENTED
      INVALID_GRAPH
      EP_FAIL
    end

    # For C API compatibility
    alias OrtErrorCode = ErrorCode

    enum OpAttrType
      UNDEFINED = 0
      INT
      INTS
      FLOAT
      FLOATS
      STRING
      STRINGS
    end

    # For C API compatibility
    alias OrtOpAttrType = OpAttrType

    enum GraphOptimizationLevel
      DISABLE_ALL     =  0
      ENABLE_BASIC    =  1
      ENABLE_EXTENDED =  2
      ENABLE_ALL      = 99
    end

    enum ExecutionMode
      SEQUENTIAL = 0
      PARALLEL   = 1
    end

    enum LanguageProjection
      C         = 0
      CPLUSPLUS
      CSHARP
      PYTHON
      JAVA
      WINML
      NODEJS
    end

    # For C API compatibility
    alias OrtLanguageProjection = LanguageProjection

    enum AllocatorType
      INVALID = -1
      DEVICE  =  0
      ARENA   =  1
    end

    # For C API compatibility
    alias OrtAllocatorType = AllocatorType

    # Memory types for allocated memory, execution provider specific types should be extended in each provider
    enum MemType
      CPU_INPUT  = -2         # Any CPU memory used by non-CPU execution provider
      CPU_OUTPUT = -1         # CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
      CPU        = CPU_OUTPUT # Temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
      DEFAULT    = 0          # The default allocator for execution provider
    end

    # For C API compatibility
    alias OrtMemType = MemType

    # This mimics OrtDevice type constants so they can be returned in the API
    enum MemoryInfoDeviceType
      CPU  = 0
      GPU
      FPGA
    end

    # For C API compatibility
    alias OrtMemoryInfoDeviceType = MemoryInfoDeviceType

    # Algorithm to use for cuDNN Convolution Op
    enum CudnnConvAlgoSearch
      EXHAUSTIVE = 0 # expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
      HEURISTIC      # lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
      DEFAULT        # default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    end

    # For C API compatibility
    alias OrtCudnnConvAlgoSearch = CudnnConvAlgoSearch

    # Runtime classes
    type OrtEnv = Void
    type OrtStatus = Void
    type OrtMemoryInfo = Void
    type OrtIoBinding = Void
    type OrtSession = Void
    type OrtValue = Void
    type OrtRunOptions = Void
    type OrtTypeInfo = Void
    type OrtTensorTypeAndShapeInfo = Void
    type OrtMapTypeInfo = Void
    type OrtSequenceTypeInfo = Void
    type OrtOptionalTypeInfo = Void
    type OrtSessionOptions = Void
    type OrtCustomOpDomain = Void
    type OrtModelMetadata = Void
    type OrtThreadPoolParams = Void
    type OrtThreadingOptions = Void
    type OrtArenaCfg = Void
    type OrtPrepackedWeightsContainer = Void
    type OrtTensorRTProviderOptionsV2 = Void
    type OrtCUDAProviderOptionsV2 = Void
    type OrtCANNProviderOptions = Void
    type OrtDnnlProviderOptions = Void
    type OrtOp = Void
    type OrtOpAttr = Void
    type OrtLogger = Void
    type OrtShapeInferContext = Void
    type OrtLoraAdapter = Void
    type OrtKernelInfo = Void
    type OrtKernelContext = Void
    type OrtCustomOp = Void
    type OrtAllocator = Void

    # Memory allocation interface
    struct OrtAllocatorStruct
      version : UInt32
      alloc : (OrtAllocator*, LibC::SizeT -> Void*)
      free : (OrtAllocator*, Void* -> Void)
      info : (OrtAllocator* -> OrtMemoryInfo*)
      reserve : (OrtAllocator*, LibC::SizeT -> Void*)
    end

    # Logging function
    alias OrtLoggingFunction = (Void*, OrtLoggingLevel, LibC::Char*, LibC::Char*, LibC::Char*, LibC::Char* -> Void)

    # Thread work loop function
    alias OrtThreadWorkerFn = (Void* -> Void)

    # Custom thread creation function
    alias OrtCustomCreateThreadFn = (Void*, OrtThreadWorkerFn, Void* -> Void*)

    # Custom thread join function
    alias OrtCustomJoinThreadFn = (Void* -> Void)

    # Callback function for RunAsync
    alias RunAsyncCallbackFn = (Void*, OrtValue**, LibC::SizeT, OrtStatus* -> Void)

    # API structure
    struct Api
      # OrtStatus
      create_status : (OrtErrorCode, LibC::Char* -> OrtStatus*)
      get_error_code : (OrtStatus* -> OrtErrorCode)
      get_error_message : (OrtStatus* -> LibC::Char*)

      # OrtEnv
      create_env : (OrtLoggingLevel, LibC::Char*, OrtEnv** -> OrtStatus*)
      create_env_with_custom_logger : (OrtLoggingFunction, Void*, OrtLoggingLevel, LibC::Char*, OrtEnv** -> OrtStatus*)
      enable_telemetry_events : (OrtEnv* -> OrtStatus*)
      disable_telemetry_events : (OrtEnv* -> OrtStatus*)

      # OrtSession
      create_session : (OrtEnv*, LibC::Char*, OrtSessionOptions*, OrtSession** -> OrtStatus*)
      create_session_from_array : (OrtEnv*, Void*, LibC::SizeT, OrtSessionOptions*, OrtSession** -> OrtStatus*)
      run : (OrtSession*, OrtRunOptions*, LibC::Char**, OrtValue**, LibC::SizeT, LibC::Char**, LibC::SizeT, OrtValue** -> OrtStatus*)

      # OrtSessionOptions
      create_session_options : (OrtSessionOptions** -> OrtStatus*)
      set_optimized_model_file_path : (OrtSessionOptions*, LibC::Char* -> OrtStatus*)
      clone_session_options : (OrtSessionOptions*, OrtSessionOptions** -> OrtStatus*)
      set_session_execution_mode : (OrtSessionOptions*, ExecutionMode -> OrtStatus*)
      enable_profiling : (OrtSessionOptions*, LibC::Char* -> OrtStatus*)
      disable_profiling : (OrtSessionOptions* -> OrtStatus*)
      enable_mem_pattern : (OrtSessionOptions* -> OrtStatus*)
      disable_mem_pattern : (OrtSessionOptions* -> OrtStatus*)
      enable_cpu_mem_arena : (OrtSessionOptions* -> OrtStatus*)
      disable_cpu_mem_arena : (OrtSessionOptions* -> OrtStatus*)
      set_session_log_id : (OrtSessionOptions*, LibC::Char* -> OrtStatus*)
      set_session_log_verbosity_level : (OrtSessionOptions*, Int32 -> OrtStatus*)
      set_session_log_severity_level : (OrtSessionOptions*, Int32 -> OrtStatus*)
      set_session_graph_optimization_level : (OrtSessionOptions*, GraphOptimizationLevel -> OrtStatus*)
      set_intra_op_num_threads : (OrtSessionOptions*, Int32 -> OrtStatus*)
      set_inter_op_num_threads : (OrtSessionOptions*, Int32 -> OrtStatus*)
      # OrtCustomOpDomain
      create_custom_op_domain : (LibC::Char*, OrtCustomOpDomain** -> OrtStatus*)
      custom_op_domain_add : (OrtCustomOpDomain*, OrtCustomOp* -> OrtStatus*)
      add_custom_op_domain : (OrtSessionOptions*, OrtCustomOpDomain* -> OrtStatus*)
      register_custom_ops_library : (OrtSessionOptions*, LibC::Char*, Void** -> OrtStatus*)

      # OrtSession
      session_get_input_count : (OrtSession*, LibC::SizeT* -> OrtStatus*)
      session_get_output_count : (OrtSession*, LibC::SizeT* -> OrtStatus*)
      session_get_overridable_initializer_count : (OrtSession*, LibC::SizeT* -> OrtStatus*)
      session_get_input_type_info : (OrtSession*, LibC::SizeT, OrtTypeInfo** -> OrtStatus*)
      session_get_output_type_info : (OrtSession*, LibC::SizeT, OrtTypeInfo** -> OrtStatus*)
      session_get_overridable_initializer_type_info : (OrtSession*, LibC::SizeT, OrtTypeInfo** -> OrtStatus*)
      session_get_input_name : (OrtSession*, LibC::SizeT, OrtAllocator*, LibC::Char** -> OrtStatus*)
      session_get_output_name : (OrtSession*, LibC::SizeT, OrtAllocator*, LibC::Char** -> OrtStatus*)
      session_get_overridable_initializer_name : (OrtSession*, LibC::SizeT, OrtAllocator*, LibC::Char** -> OrtStatus*)

      # OrtRunOptions
      create_run_options : (OrtRunOptions** -> OrtStatus*)
      run_options_set_run_log_verbosity_level : (OrtRunOptions*, Int32 -> OrtStatus*)
      run_options_set_run_log_severity_level : (OrtRunOptions*, Int32 -> OrtStatus*)
      run_options_set_run_tag : (OrtRunOptions*, LibC::Char* -> OrtStatus*)
      run_options_get_run_log_verbosity_level : (OrtRunOptions*, Int32* -> OrtStatus*)
      run_options_get_run_log_severity_level : (OrtRunOptions*, Int32* -> OrtStatus*)
      run_options_get_run_tag : (OrtRunOptions*, LibC::Char** -> OrtStatus*)
      run_options_set_terminate : (OrtRunOptions* -> OrtStatus*)
      run_options_unset_terminate : (OrtRunOptions* -> OrtStatus*)
      # OrtValue - Tensor
      create_tensor_as_ort_value : (OrtAllocator*, Int64*, LibC::SizeT, ONNXTensorElementDataType, OrtValue** -> OrtStatus*)
      create_tensor_with_data_as_ort_value : (OrtMemoryInfo*, Void*, LibC::SizeT, Int64*, LibC::SizeT, ONNXTensorElementDataType, OrtValue** -> OrtStatus*)
      is_tensor : (OrtValue*, LibC::Int* -> OrtStatus*)
      get_tensor_mutable_data : (OrtValue*, Void** -> OrtStatus*)
      fill_string_tensor : (OrtValue*, LibC::Char**, LibC::SizeT -> OrtStatus*)
      get_string_tensor_data_length : (OrtValue*, LibC::SizeT* -> OrtStatus*)
      get_string_tensor_content : (OrtValue*, Void*, LibC::SizeT, LibC::SizeT*, LibC::SizeT -> OrtStatus*)

      # OrtTypeInfo
      cast_type_info_to_tensor_info : (OrtTypeInfo*, OrtTensorTypeAndShapeInfo** -> OrtStatus*)
      get_onnx_type_from_type_info : (OrtTypeInfo*, ONNXType* -> OrtStatus*)

      # OrtTensorTypeAndShapeInfo
      create_tensor_type_and_shape_info : (OrtTensorTypeAndShapeInfo** -> OrtStatus*)
      set_tensor_element_type : (OrtTensorTypeAndShapeInfo*, ONNXTensorElementDataType -> OrtStatus*)
      set_dimensions : (OrtTensorTypeAndShapeInfo*, Int64*, LibC::SizeT -> OrtStatus*)
      get_tensor_element_type : (OrtTensorTypeAndShapeInfo*, ONNXTensorElementDataType* -> OrtStatus*)
      get_dimensions_count : (OrtTensorTypeAndShapeInfo*, LibC::SizeT* -> OrtStatus*)
      get_dimensions : (OrtTensorTypeAndShapeInfo*, Int64*, LibC::SizeT -> OrtStatus*)
      get_symbolic_dimensions : (OrtTensorTypeAndShapeInfo*, LibC::Char**, LibC::SizeT -> OrtStatus*)
      get_tensor_shape_element_count : (OrtTensorTypeAndShapeInfo*, LibC::SizeT* -> OrtStatus*)
      get_tensor_type_and_shape : (OrtValue*, OrtTensorTypeAndShapeInfo** -> OrtStatus*)
      get_type_info : (OrtValue*, OrtTypeInfo** -> OrtStatus*)
      get_value_type : (OrtValue*, ONNXType* -> OrtStatus*)
      # OrtMemoryInfo
      create_memory_info : (LibC::Char*, OrtAllocatorType, Int32, OrtMemType, OrtMemoryInfo** -> OrtStatus*)
      create_cpu_memory_info : (OrtAllocatorType, OrtMemType, OrtMemoryInfo** -> OrtStatus*)
      compare_memory_info : (OrtMemoryInfo*, OrtMemoryInfo*, LibC::Int* -> OrtStatus*)
      memory_info_get_name : (OrtMemoryInfo*, LibC::Char** -> OrtStatus*)
      memory_info_get_id : (OrtMemoryInfo*, LibC::Int* -> OrtStatus*)
      memory_info_get_mem_type : (OrtMemoryInfo*, OrtMemType* -> OrtStatus*)
      memory_info_get_type : (OrtMemoryInfo*, OrtAllocatorType* -> OrtStatus*)

      # OrtAllocator
      allocator_alloc : (OrtAllocator*, LibC::SizeT, Void** -> OrtStatus*)
      allocator_free : (OrtAllocator*, Void* -> OrtStatus*)
      allocator_get_info : (OrtAllocator*, OrtMemoryInfo** -> OrtStatus*)
      get_allocator_with_default_options : (OrtAllocator** -> OrtStatus*)

      # OrtSessionOptions
      add_free_dimension_override : (OrtSessionOptions*, LibC::Char*, Int64 -> OrtStatus*)

      # OrtValue
      get_value : (OrtValue*, Int32, OrtAllocator*, OrtValue** -> OrtStatus*)
      get_value_count : (OrtValue*, LibC::SizeT* -> OrtStatus*)
      create_value : (OrtValue**, LibC::SizeT, ONNXType, OrtValue** -> OrtStatus*)
      create_opaque_value : (LibC::Char*, LibC::Char*, Void*, LibC::SizeT, OrtValue** -> OrtStatus*)
      get_opaque_value : (LibC::Char*, LibC::Char*, OrtValue*, Void*, LibC::SizeT -> OrtStatus*)
      # OrtKernelInfo - Custom operator APIs
      kernel_info_get_attribute_float : (OrtKernelInfo*, LibC::Char*, Float32* -> OrtStatus*)
      kernel_info_get_attribute_int64 : (OrtKernelInfo*, LibC::Char*, Int64* -> OrtStatus*)
      kernel_info_get_attribute_string : (OrtKernelInfo*, LibC::Char*, LibC::Char*, LibC::SizeT* -> OrtStatus*)

      # OrtKernelContext - Custom operator APIs
      kernel_context_get_input_count : (OrtKernelContext*, LibC::SizeT* -> OrtStatus*)
      kernel_context_get_output_count : (OrtKernelContext*, LibC::SizeT* -> OrtStatus*)
      kernel_context_get_input : (OrtKernelContext*, LibC::SizeT, OrtValue** -> OrtStatus*)
      kernel_context_get_output : (OrtKernelContext*, LibC::SizeT, Int64*, LibC::SizeT, OrtValue** -> OrtStatus*)
      # Release functions
      release_env : (OrtEnv* -> Void)
      release_status : (OrtStatus* -> Void)
      release_memory_info : (OrtMemoryInfo* -> Void)
      release_session : (OrtSession* -> Void)
      release_value : (OrtValue* -> Void)
      release_run_options : (OrtRunOptions* -> Void)
      release_type_info : (OrtTypeInfo* -> Void)
      release_tensor_type_and_shape_info : (OrtTensorTypeAndShapeInfo* -> Void)
      release_session_options : (OrtSessionOptions* -> Void)
      release_custom_op_domain : (OrtCustomOpDomain* -> Void)
      # OrtTypeInfo
      get_denotation_from_type_info : (OrtTypeInfo*, LibC::Char**, LibC::SizeT* -> OrtStatus*)
      cast_type_info_to_map_type_info : (OrtTypeInfo*, OrtMapTypeInfo** -> OrtStatus*)
      cast_type_info_to_sequence_type_info : (OrtTypeInfo*, OrtSequenceTypeInfo** -> OrtStatus*)

      # OrtMapTypeInfo
      get_map_key_type : (OrtMapTypeInfo*, ONNXTensorElementDataType* -> OrtStatus*)
      get_map_value_type : (OrtMapTypeInfo*, OrtTypeInfo** -> OrtStatus*)

      # OrtSequenceTypeInfo
      get_sequence_element_type : (OrtSequenceTypeInfo*, OrtTypeInfo** -> OrtStatus*)

      # Release functions
      release_map_type_info : (OrtMapTypeInfo* -> Void)
      release_sequence_type_info : (OrtSequenceTypeInfo* -> Void)
      # OrtSession
      session_end_profiling : (OrtSession*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      session_get_model_metadata : (OrtSession*, OrtModelMetadata** -> OrtStatus*)

      # OrtModelMetadata
      model_metadata_get_producer_name : (OrtModelMetadata*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      model_metadata_get_graph_name : (OrtModelMetadata*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      model_metadata_get_domain : (OrtModelMetadata*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      model_metadata_get_description : (OrtModelMetadata*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      model_metadata_lookup_custom_metadata_map : (OrtModelMetadata*, OrtAllocator*, LibC::Char*, LibC::Char** -> OrtStatus*)
      model_metadata_get_version : (OrtModelMetadata*, Int64* -> OrtStatus*)

      # Release functions
      release_model_metadata : (OrtModelMetadata* -> Void)
      # OrtEnv - Threading
      create_env_with_global_thread_pools : (OrtLoggingLevel, LibC::Char*, OrtThreadingOptions*, OrtEnv** -> OrtStatus*)
      disable_per_session_threads : (OrtSessionOptions* -> OrtStatus*)
      create_threading_options : (OrtThreadingOptions** -> OrtStatus*)
      release_threading_options : (OrtThreadingOptions* -> Void)

      # OrtModelMetadata
      model_metadata_get_custom_metadata_map_keys : (OrtModelMetadata*, OrtAllocator*, LibC::Char***, LibC::SizeT* -> OrtStatus*)

      # OrtSessionOptions
      add_free_dimension_override_by_name : (OrtSessionOptions*, LibC::Char*, Int64 -> OrtStatus*)

      # Execution Providers
      get_available_providers : (LibC::Char***, LibC::Int* -> OrtStatus*)
      release_available_providers : (LibC::Char**, LibC::Int -> Void)
      # OrtValue - String Tensor
      get_string_tensor_element_length : (OrtValue*, LibC::SizeT, LibC::SizeT* -> OrtStatus*)
      get_string_tensor_element : (OrtValue*, LibC::SizeT, LibC::Char*, LibC::SizeT -> OrtStatus*)
      fill_string_tensor_element : (OrtValue*, LibC::SizeT, LibC::Char* -> OrtStatus*)

      # OrtSessionOptions
      add_session_config_entry : (OrtSessionOptions*, LibC::Char*, LibC::Char* -> OrtStatus*)
      # OrtAllocator
      create_allocator : (OrtSession*, OrtMemoryInfo*, OrtAllocator** -> OrtStatus*)
      release_allocator : (OrtAllocator* -> Void)

      # OrtIoBinding
      run_with_binding : (OrtSession*, OrtRunOptions*, OrtIoBinding* -> OrtStatus*)
      create_io_binding : (OrtSession*, OrtIoBinding** -> OrtStatus*)
      release_io_binding : (OrtIoBinding* -> Void)
      bind_input : (OrtIoBinding*, LibC::Char*, OrtValue* -> OrtStatus*)
      bind_output : (OrtIoBinding*, LibC::Char*, OrtValue* -> OrtStatus*)
      bind_output_to_device : (OrtIoBinding*, LibC::Char*, OrtMemoryInfo* -> OrtStatus*)
      get_bound_output_names : (OrtIoBinding*, OrtAllocator*, LibC::Char***, LibC::SizeT* -> OrtStatus*)
      get_bound_output_values : (OrtIoBinding*, OrtAllocator*, OrtValue***, LibC::SizeT* -> OrtStatus*)
      clear_bound_inputs : (OrtIoBinding* -> OrtStatus*)
      clear_bound_outputs : (OrtIoBinding* -> OrtStatus*)
      # OrtValue
      tensor_at : (OrtValue*, Int64*, LibC::SizeT, OrtValue** -> OrtStatus*)

      # OrtAllocator
      create_and_register_allocator : (OrtEnv*, OrtAllocatorStruct* -> OrtStatus*)

      # OrtEnv
      set_language_projection : (OrtEnv*, OrtLanguageProjection -> OrtStatus*)
      session_get_profiling_start_time_ns : (OrtSession*, UInt64* -> OrtStatus*)
      set_global_intra_op_num_threads : (OrtEnv*, Int32 -> OrtStatus*)
      set_global_inter_op_num_threads : (OrtEnv*, Int32 -> OrtStatus*)
      set_global_spin_control : (OrtEnv*, Int32 -> OrtStatus*)

      # OrtSessionOptions
      add_initializer : (OrtSessionOptions*, LibC::Char*, OrtValue* -> OrtStatus*)

      # OrtEnv
      create_env_with_custom_logger_and_global_thread_pools : (OrtLoggingFunction, Void*, OrtLoggingLevel, LibC::Char*, OrtThreadingOptions*, OrtEnv** -> OrtStatus*)
      # Execution Providers
      session_options_append_execution_provider_cuda : (OrtSessionOptions*, Int32 -> OrtStatus*)
      session_options_append_execution_provider_rocm : (OrtSessionOptions*, Int32 -> OrtStatus*)
      session_options_append_execution_provider_open_vino : (OrtSessionOptions*, LibC::Char* -> OrtStatus*)

      # OrtEnv
      set_global_denormal_as_zero : (OrtEnv* -> OrtStatus*)

      # OrtArenaCfg
      create_arena_cfg : (LibC::SizeT, Int32, Int32, Int32, OrtArenaCfg** -> OrtStatus*)
      release_arena_cfg : (OrtArenaCfg* -> Void)

      # OrtModelMetadata
      model_metadata_get_graph_description : (OrtModelMetadata*, OrtAllocator*, LibC::Char** -> OrtStatus*)

      # Execution Providers
      session_options_append_execution_provider_tensor_rt : (OrtSessionOptions*, Int32 -> OrtStatus*)
      set_current_gpu_device_id : (Int32 -> OrtStatus*)
      get_current_gpu_device_id : (Int32* -> OrtStatus*)
      # OrtKernelInfo - Custom operator APIs
      kernel_info_get_attribute_array_float : (OrtKernelInfo*, LibC::Char*, Float32*, LibC::SizeT* -> OrtStatus*)
      kernel_info_get_attribute_array_int64 : (OrtKernelInfo*, LibC::Char*, Int64*, LibC::SizeT* -> OrtStatus*)

      # OrtArenaCfg
      create_arena_cfg_v2 : (LibC::SizeT, Int32, Int32, Int32, Int32, Int32, OrtArenaCfg** -> OrtStatus*)

      # OrtRunOptions
      add_run_config_entry : (OrtRunOptions*, LibC::Char*, LibC::Char* -> OrtStatus*)

      # OrtPrepackedWeightsContainer
      create_prepacked_weights_container : (OrtPrepackedWeightsContainer** -> OrtStatus*)
      release_prepacked_weights_container : (OrtPrepackedWeightsContainer* -> Void)

      # OrtSession
      create_session_with_prepacked_weights_container : (OrtEnv*, LibC::Char*, OrtSessionOptions*, OrtPrepackedWeightsContainer*, OrtSession** -> OrtStatus*)
      create_session_from_array_with_prepacked_weights_container : (OrtEnv*, Void*, LibC::SizeT, OrtSessionOptions*, OrtPrepackedWeightsContainer*, OrtSession** -> OrtStatus*)
      # Execution Providers - TensorRT
      session_options_append_execution_provider_tensor_rt_v2 : (OrtSessionOptions*, OrtTensorRTProviderOptionsV2* -> OrtStatus*)
      create_tensor_rt_provider_options : (OrtTensorRTProviderOptionsV2** -> OrtStatus*)
      update_tensor_rt_provider_options : (OrtTensorRTProviderOptionsV2*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      get_tensor_rt_provider_options_as_string : (OrtTensorRTProviderOptionsV2*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      release_tensor_rt_provider_options : (OrtTensorRTProviderOptionsV2* -> Void)

      # OrtSessionOptions
      enable_ort_custom_ops : (OrtSessionOptions* -> OrtStatus*)

      # OrtAllocator
      register_allocator : (OrtEnv*, OrtAllocator* -> OrtStatus*)
      unregister_allocator : (OrtEnv*, OrtAllocator* -> OrtStatus*)
      # OrtValue - Sparse Tensor
      is_sparse_tensor : (OrtValue*, LibC::Int* -> OrtStatus*)
      create_sparse_tensor_as_ort_value : (OrtAllocator*, Int64*, LibC::SizeT, ONNXTensorElementDataType, OrtValue** -> OrtStatus*)
      fill_sparse_tensor_coo : (OrtValue*, OrtMemoryInfo*, Int64*, LibC::SizeT, Void*, Int64*, LibC::SizeT -> OrtStatus*)
      fill_sparse_tensor_csr : (OrtValue*, OrtMemoryInfo*, Int64*, LibC::SizeT, Void*, Int64*, LibC::SizeT, Int64*, LibC::SizeT -> OrtStatus*)
      fill_sparse_tensor_block_sparse : (OrtValue*, OrtMemoryInfo*, Int64*, LibC::SizeT, Void*, Int64*, LibC::SizeT, Int32* -> OrtStatus*)
      create_sparse_tensor_with_values_as_ort_value : (OrtMemoryInfo*, Void*, LibC::SizeT, Int64*, LibC::SizeT, ONNXTensorElementDataType, OrtValue** -> OrtStatus*)
      use_coo_indices : (OrtValue*, Int64*, LibC::SizeT -> OrtStatus*)
      use_csr_indices : (OrtValue*, Int64*, LibC::SizeT, Int64*, LibC::SizeT -> OrtStatus*)
      use_block_sparse_indices : (OrtValue*, Int64*, LibC::SizeT, Int64*, LibC::SizeT -> OrtStatus*)
      get_sparse_tensor_format : (OrtValue*, OrtSparseFormat* -> OrtStatus*)
      get_sparse_tensor_values_type_and_shape : (OrtValue*, OrtTensorTypeAndShapeInfo** -> OrtStatus*)
      get_sparse_tensor_values : (OrtValue*, OrtValue** -> OrtStatus*)
      get_sparse_tensor_indices_type_shape : (OrtValue*, OrtSparseIndicesFormat, OrtTensorTypeAndShapeInfo** -> OrtStatus*)
      get_sparse_tensor_indices : (OrtValue*, OrtSparseIndicesFormat, OrtValue** -> OrtStatus*)

      # OrtValue
      has_value : (OrtValue*, LibC::Int* -> OrtStatus*)

      # OrtKernelContext
      kernel_context_get_gpu_compute_stream : (OrtKernelContext*, Void** -> OrtStatus*)

      # OrtValue
      get_tensor_memory_info : (OrtValue*, OrtMemoryInfo** -> OrtStatus*)

      # Execution Providers
      get_execution_provider_api : (LibC::Char*, Int32, Void** -> OrtStatus*)
      # OrtSessionOptions - Custom thread functions
      session_options_set_custom_create_thread_fn : (OrtSessionOptions*, OrtCustomCreateThreadFn, Void* -> OrtStatus*)
      session_options_set_custom_thread_creation_options : (OrtSessionOptions*, Void* -> OrtStatus*)
      session_options_set_custom_join_thread_fn : (OrtSessionOptions*, OrtCustomJoinThreadFn -> OrtStatus*)

      # OrtEnv - Custom thread functions
      set_global_custom_create_thread_fn : (OrtEnv*, OrtCustomCreateThreadFn, Void* -> OrtStatus*)
      set_global_custom_thread_creation_options : (OrtEnv*, Void* -> OrtStatus*)
      set_global_custom_join_thread_fn : (OrtEnv*, OrtCustomJoinThreadFn -> OrtStatus*)

      # OrtIoBinding
      synchronize_bound_inputs : (OrtIoBinding* -> OrtStatus*)
      synchronize_bound_outputs : (OrtIoBinding* -> OrtStatus*)

      # Execution Providers - CUDA
      session_options_append_execution_provider_cuda_v2 : (OrtSessionOptions*, OrtCUDAProviderOptionsV2* -> OrtStatus*)
      create_cuda_provider_options : (OrtCUDAProviderOptionsV2** -> OrtStatus*)
      update_cuda_provider_options : (OrtCUDAProviderOptionsV2*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      get_cuda_provider_options_as_string : (OrtCUDAProviderOptionsV2*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      release_cuda_provider_options : (OrtCUDAProviderOptionsV2* -> Void)

      # Execution Providers - MIGraphX
      session_options_append_execution_provider_mi_graph_x : (OrtSessionOptions*, Int32, LibC::Char* -> OrtStatus*)
    end

    struct ApiBase
      get_api : (UInt32 -> Api*)
      get_version_string : (-> LibC::Char*)
    end

    fun OrtGetApiBase : ApiBase*
  end
end
