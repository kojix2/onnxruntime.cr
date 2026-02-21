module OnnxRuntime
  @[Link("onnxruntime")]
  lib LibOnnxRuntime
    ORT_API_VERSION = 24_u32

    {% if flag?(:win32) %}
      alias ORTCHAR_T = LibC::WCHAR
    {% else %}
      alias ORTCHAR_T = LibC::Char
    {% end %}

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
      # Float4 types were introduced in ONNX 1.18
      FLOAT4E2M1 # maps to a pair of packed float4 values (size == 1 byte)
      # Int2 types were introduced in ONNX 1.20
      UINT2 # maps to 4 packed uint2 values (size == 1 byte)
      INT2  # maps to 4 packed int2 values (size == 1 byte)
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
      OK                         = 0
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
      MODEL_LOAD_CANCELED
      MODEL_REQUIRES_COMPILATION
      NOT_FOUND
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
      GRAPH
      TENSOR
    end

    # For C API compatibility
    alias OrtOpAttrType = OpAttrType

    enum GraphOptimizationLevel
      DISABLE_ALL     =  0
      ENABLE_BASIC    =  1
      ENABLE_EXTENDED =  2
      ENABLE_LAYOUT   =  3
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
      INVALID   = -1
      DEVICE    =  0
      ARENA     =  1
      READ_ONLY =  2
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
      NPU
    end

    # This matches OrtDevice::MemoryType values
    enum DeviceMemoryType
      DEFAULT         = 0
      HOST_ACCESSIBLE = 5
    end

    # For C API compatibility
    alias OrtDeviceMemoryType = DeviceMemoryType

    enum HardwareDeviceType
      CPU
      GPU
      NPU
    end

    # For C API compatibility
    alias OrtHardwareDeviceType = HardwareDeviceType

    enum ExecutionProviderDevicePolicy
      DEFAULT
      PREFER_CPU
      PREFER_NPU
      PREFER_GPU
      MAX_PERFORMANCE
      MAX_EFFICIENCY
      MIN_OVERALL_POWER
    end

    # For C API compatibility
    alias OrtExecutionProviderDevicePolicy = ExecutionProviderDevicePolicy

    enum DeviceEpIncompatibilityReason
      NONE                = 0
      DRIVER_INCOMPATIBLE = 1 << 0
      DEVICE_INCOMPATIBLE = 1 << 1
      MISSING_DEPENDENCY  = 1 << 2
      UNKNOWN             = 1 << 31
    end

    # For C API compatibility
    alias OrtDeviceEpIncompatibilityReason = DeviceEpIncompatibilityReason

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

    # External memory handle type for importing GPU resources
    enum ExternalMemoryHandleType
      D3D12_RESOURCE = 0
      D3D12_HEAP     = 1
    end

    # For C API compatibility
    alias OrtExternalMemoryHandleType = ExternalMemoryHandleType

    # External semaphore type for GPU synchronization
    enum ExternalSemaphoreType
      D3D12_FENCE = 0
    end

    # For C API compatibility
    alias OrtExternalSemaphoreType = ExternalSemaphoreType

    enum CompiledModelCompatibility
      EP_NOT_APPLICABLE                 = 0
      EP_SUPPORTED_OPTIMAL
      EP_SUPPORTED_PREFER_RECOMPILATION
      EP_UNSUPPORTED
    end

    # For C API compatibility
    alias OrtCompiledModelCompatibility = CompiledModelCompatibility

    enum CompileApiFlags
      NONE                        = 0
      ERROR_IF_NO_NODES_COMPILED  = 1 << 0
      ERROR_IF_OUTPUT_FILE_EXISTS = 1 << 1
    end

    # For C API compatibility
    alias OrtCompileApiFlags = CompileApiFlags

    struct OrtExternalMemoryDescriptor
      version : UInt32
      handle_type : OrtExternalMemoryHandleType
      native_handle : Void*
      size_bytes : LibC::SizeT
      offset_bytes : LibC::SizeT
    end

    struct OrtExternalSemaphoreDescriptor
      version : UInt32
      type : OrtExternalSemaphoreType
      native_handle : Void*
    end

    struct OrtExternalTensorDescriptor
      version : UInt32
      element_type : ONNXTensorElementDataType
      shape : Int64*
      rank : LibC::SizeT
      offset_bytes : LibC::SizeT
    end

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
    type OrtTrainingApi = Void
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
    type OrtValueInfo = Void
    type OrtNode = Void
    type OrtGraph = Void
    type OrtModel = Void
    type OrtModelCompilationOptions = Void
    type OrtHardwareDevice = Void
    type OrtEpDevice = Void
    type OrtKeyValuePairs = Void
    type OrtSyncStream = Void
    type OrtExternalInitializerInfo = Void
    type OrtExternalResourceImporter = Void
    type OrtExternalMemoryHandle = Void
    type OrtExternalSemaphoreHandle = Void
    type OrtDeviceEpIncompatibilityDetails = Void
    type OrtEpAssignedSubgraph = Void
    type OrtEpAssignedNode = Void
    type OrtEpApi = Void
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

    # Write callback for output model serialization
    alias OrtWriteBufferFunc = (Void*, Void*, LibC::SizeT -> OrtStatus*)

    # Initializer location callback
    alias OrtGetInitializerLocationFunc = (Void*, LibC::Char*, OrtValue*, OrtExternalInitializerInfo*, OrtExternalInitializerInfo** -> OrtStatus*)

    # Execution provider selection delegate
    alias EpSelectionDelegate = (OrtEpDevice**, LibC::SizeT, OrtKeyValuePairs*, OrtKeyValuePairs*, OrtEpDevice**, LibC::SizeT, LibC::SizeT*, Void* -> OrtStatus*)

    struct EnvCreationOptions
      version : UInt32
      logging_severity_level : Int32
      log_id : LibC::Char*
      custom_logging_function : OrtLoggingFunction
      custom_logging_param : Void*
      threading_options : OrtThreadingOptions*
      config_entries : OrtKeyValuePairs*
    end

    struct OrtCUDAProviderOptions
      device_id : Int32
      cudnn_conv_algo_search : OrtCudnnConvAlgoSearch
      gpu_mem_limit : LibC::SizeT
      arena_extend_strategy : Int32
      do_copy_in_default_stream : Int32
      has_user_compute_stream : Int32
      user_compute_stream : Void*
      default_memory_arena_cfg : OrtArenaCfg*
      tunable_op_enable : Int32
      tunable_op_tuning_enable : Int32
      tunable_op_max_tuning_duration_ms : Int32
    end

    struct OrtROCMProviderOptions
      device_id : Int32
      miopen_conv_exhaustive_search : Int32
      gpu_mem_limit : LibC::SizeT
      arena_extend_strategy : Int32
      do_copy_in_default_stream : Int32
      has_user_compute_stream : Int32
      user_compute_stream : Void*
      default_memory_arena_cfg : OrtArenaCfg*
      enable_hip_graph : Int32
      tunable_op_enable : Int32
      tunable_op_tuning_enable : Int32
      tunable_op_max_tuning_duration_ms : Int32
    end

    struct OrtTensorRTProviderOptions
      device_id : Int32
      has_user_compute_stream : Int32
      user_compute_stream : Void*
      trt_max_partition_iterations : Int32
      trt_min_subgraph_size : Int32
      trt_max_workspace_size : LibC::SizeT
      trt_fp16_enable : Int32
      trt_int8_enable : Int32
      trt_int8_calibration_table_name : LibC::Char*
      trt_int8_use_native_calibration_table : Int32
      trt_dla_enable : Int32
      trt_dla_core : Int32
      trt_dump_subgraphs : Int32
      trt_engine_cache_enable : Int32
      trt_engine_cache_path : LibC::Char*
      trt_engine_decryption_enable : Int32
      trt_engine_decryption_lib_path : LibC::Char*
      trt_force_sequential_engine_build : Int32
    end

    struct OrtMIGraphXProviderOptions
      device_id : Int32
      migraphx_fp16_enable : Int32
      migraphx_fp8_enable : Int32
      migraphx_int8_enable : Int32
      migraphx_use_native_calibration_table : Int32
      migraphx_int8_calibration_table_name : LibC::Char*
      migraphx_save_compiled_model : Int32
      migraphx_save_model_path : LibC::Char*
      migraphx_load_compiled_model : Int32
      migraphx_load_model_path : LibC::Char*
      migraphx_exhaustive_tune : Bool
      migraphx_mem_limit : LibC::SizeT
      migraphx_arena_extend_strategy : Int32
    end

    struct OrtOpenVINOProviderOptions
      device_type : LibC::Char*
      enable_npu_fast_compile : UInt8
      device_id : LibC::Char*
      num_of_threads : LibC::SizeT
      cache_dir : LibC::Char*
      context : Void*
      enable_opencl_throttling : UInt8
      enable_dynamic_shapes : UInt8
    end

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
      create_status : (OrtErrorCode, LibC::Char* -> OrtStatus*)
      get_error_code : (OrtStatus* -> OrtErrorCode)
      get_error_message : (OrtStatus* -> LibC::Char*)

      create_env : (OrtLoggingLevel, LibC::Char*, OrtEnv** -> OrtStatus*)
      create_env_with_custom_logger : (OrtLoggingFunction, Void*, OrtLoggingLevel, LibC::Char*, OrtEnv** -> OrtStatus*)
      enable_telemetry_events : (OrtEnv* -> OrtStatus*)
      disable_telemetry_events : (OrtEnv* -> OrtStatus*)

      create_session : (OrtEnv*, ORTCHAR_T*, OrtSessionOptions*, OrtSession** -> OrtStatus*)
      create_session_from_array : (OrtEnv*, Void*, LibC::SizeT, OrtSessionOptions*, OrtSession** -> OrtStatus*)
      run : (OrtSession*, OrtRunOptions*, LibC::Char**, OrtValue**, LibC::SizeT, LibC::Char**, LibC::SizeT, OrtValue** -> OrtStatus*)

      create_session_options : (OrtSessionOptions** -> OrtStatus*)
      set_optimized_model_file_path : (OrtSessionOptions*, ORTCHAR_T* -> OrtStatus*)
      clone_session_options : (OrtSessionOptions*, OrtSessionOptions** -> OrtStatus*)
      set_session_execution_mode : (OrtSessionOptions*, ExecutionMode -> OrtStatus*)
      enable_profiling : (OrtSessionOptions*, ORTCHAR_T* -> OrtStatus*)
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

      create_custom_op_domain : (LibC::Char*, OrtCustomOpDomain** -> OrtStatus*)
      custom_op_domain_add : (OrtCustomOpDomain*, OrtCustomOp* -> OrtStatus*)

      add_custom_op_domain : (OrtSessionOptions*, OrtCustomOpDomain* -> OrtStatus*)
      register_custom_ops_library : (OrtSessionOptions*, LibC::Char*, Void** -> OrtStatus*)

      session_get_input_count : (OrtSession*, LibC::SizeT* -> OrtStatus*)
      session_get_output_count : (OrtSession*, LibC::SizeT* -> OrtStatus*)
      session_get_overridable_initializer_count : (OrtSession*, LibC::SizeT* -> OrtStatus*)
      session_get_input_type_info : (OrtSession*, LibC::SizeT, OrtTypeInfo** -> OrtStatus*)
      session_get_output_type_info : (OrtSession*, LibC::SizeT, OrtTypeInfo** -> OrtStatus*)
      session_get_overridable_initializer_type_info : (OrtSession*, LibC::SizeT, OrtTypeInfo** -> OrtStatus*)
      session_get_input_name : (OrtSession*, LibC::SizeT, OrtAllocator*, LibC::Char** -> OrtStatus*)
      session_get_output_name : (OrtSession*, LibC::SizeT, OrtAllocator*, LibC::Char** -> OrtStatus*)
      session_get_overridable_initializer_name : (OrtSession*, LibC::SizeT, OrtAllocator*, LibC::Char** -> OrtStatus*)

      create_run_options : (OrtRunOptions** -> OrtStatus*)
      run_options_set_run_log_verbosity_level : (OrtRunOptions*, Int32 -> OrtStatus*)
      run_options_set_run_log_severity_level : (OrtRunOptions*, Int32 -> OrtStatus*)
      run_options_set_run_tag : (OrtRunOptions*, LibC::Char* -> OrtStatus*)
      run_options_get_run_log_verbosity_level : (OrtRunOptions*, Int32* -> OrtStatus*)
      run_options_get_run_log_severity_level : (OrtRunOptions*, Int32* -> OrtStatus*)
      run_options_get_run_tag : (OrtRunOptions*, LibC::Char** -> OrtStatus*)
      run_options_set_terminate : (OrtRunOptions* -> OrtStatus*)
      run_options_unset_terminate : (OrtRunOptions* -> OrtStatus*)

      create_tensor_as_ort_value : (OrtAllocator*, Int64*, LibC::SizeT, ONNXTensorElementDataType, OrtValue** -> OrtStatus*)
      create_tensor_with_data_as_ort_value : (OrtMemoryInfo*, Void*, LibC::SizeT, Int64*, LibC::SizeT, ONNXTensorElementDataType, OrtValue** -> OrtStatus*)
      is_tensor : (OrtValue*, Int32* -> OrtStatus*)
      get_tensor_mutable_data : (OrtValue*, Void** -> OrtStatus*)
      fill_string_tensor : (OrtValue*, LibC::Char**, LibC::SizeT -> OrtStatus*)
      get_string_tensor_data_length : (OrtValue*, LibC::SizeT* -> OrtStatus*)
      get_string_tensor_content : (OrtValue*, Void*, LibC::SizeT, LibC::SizeT*, LibC::SizeT -> OrtStatus*)

      cast_type_info_to_tensor_info : (OrtTypeInfo*, OrtTensorTypeAndShapeInfo** -> OrtStatus*)
      get_onnx_type_from_type_info : (OrtTypeInfo*, ONNXType* -> OrtStatus*)

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

      create_memory_info : (LibC::Char*, OrtAllocatorType, Int32, OrtMemType, OrtMemoryInfo** -> OrtStatus*)
      create_cpu_memory_info : (OrtAllocatorType, OrtMemType, OrtMemoryInfo** -> OrtStatus*)
      compare_memory_info : (OrtMemoryInfo*, OrtMemoryInfo*, Int32* -> OrtStatus*)
      memory_info_get_name : (OrtMemoryInfo*, LibC::Char** -> OrtStatus*)
      memory_info_get_id : (OrtMemoryInfo*, Int32* -> OrtStatus*)
      memory_info_get_mem_type : (OrtMemoryInfo*, OrtMemType* -> OrtStatus*)
      memory_info_get_type : (OrtMemoryInfo*, OrtAllocatorType* -> OrtStatus*)

      allocator_alloc : (OrtAllocator*, LibC::SizeT, Void** -> OrtStatus*)
      allocator_free : (OrtAllocator*, Void* -> OrtStatus*)
      allocator_get_info : (OrtAllocator*, OrtMemoryInfo** -> OrtStatus*)
      get_allocator_with_default_options : (OrtAllocator** -> OrtStatus*)

      add_free_dimension_override : (OrtSessionOptions*, LibC::Char*, Int64 -> OrtStatus*)

      get_value : (OrtValue*, Int32, OrtAllocator*, OrtValue** -> OrtStatus*)
      get_value_count : (OrtValue*, LibC::SizeT* -> OrtStatus*)
      create_value : (OrtValue**, LibC::SizeT, ONNXType, OrtValue** -> OrtStatus*)
      create_opaque_value : (LibC::Char*, LibC::Char*, Void*, LibC::SizeT, OrtValue** -> OrtStatus*)
      get_opaque_value : (LibC::Char*, LibC::Char*, OrtValue*, Void*, LibC::SizeT -> OrtStatus*)

      kernel_info_get_attribute_float : (OrtKernelInfo*, LibC::Char*, Float32* -> OrtStatus*)
      kernel_info_get_attribute_int64 : (OrtKernelInfo*, LibC::Char*, Int64* -> OrtStatus*)
      kernel_info_get_attribute_string : (OrtKernelInfo*, LibC::Char*, LibC::Char*, LibC::SizeT* -> OrtStatus*)

      kernel_context_get_input_count : (OrtKernelContext*, LibC::SizeT* -> OrtStatus*)
      kernel_context_get_output_count : (OrtKernelContext*, LibC::SizeT* -> OrtStatus*)
      kernel_context_get_input : (OrtKernelContext*, LibC::SizeT, OrtValue** -> OrtStatus*)
      kernel_context_get_output : (OrtKernelContext*, LibC::SizeT, Int64*, LibC::SizeT, OrtValue** -> OrtStatus*)

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

      get_denotation_from_type_info : (OrtTypeInfo*, LibC::Char**, LibC::SizeT* -> OrtStatus*)
      cast_type_info_to_map_type_info : (OrtTypeInfo*, OrtMapTypeInfo** -> OrtStatus*)
      cast_type_info_to_sequence_type_info : (OrtTypeInfo*, OrtSequenceTypeInfo** -> OrtStatus*)

      get_map_key_type : (OrtMapTypeInfo*, ONNXTensorElementDataType* -> OrtStatus*)
      get_map_value_type : (OrtMapTypeInfo*, OrtTypeInfo** -> OrtStatus*)

      get_sequence_element_type : (OrtSequenceTypeInfo*, OrtTypeInfo** -> OrtStatus*)

      release_map_type_info : (OrtMapTypeInfo* -> Void)
      release_sequence_type_info : (OrtSequenceTypeInfo* -> Void)

      session_end_profiling : (OrtSession*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      session_get_model_metadata : (OrtSession*, OrtModelMetadata** -> OrtStatus*)

      model_metadata_get_producer_name : (OrtModelMetadata*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      model_metadata_get_graph_name : (OrtModelMetadata*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      model_metadata_get_domain : (OrtModelMetadata*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      model_metadata_get_description : (OrtModelMetadata*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      model_metadata_lookup_custom_metadata_map : (OrtModelMetadata*, OrtAllocator*, LibC::Char*, LibC::Char** -> OrtStatus*)
      model_metadata_get_version : (OrtModelMetadata*, Int64* -> OrtStatus*)
      release_model_metadata : (OrtModelMetadata* -> Void)

      create_env_with_global_thread_pools : (OrtLoggingLevel, LibC::Char*, OrtThreadingOptions*, OrtEnv** -> OrtStatus*)
      disable_per_session_threads : (OrtSessionOptions* -> OrtStatus*)

      create_threading_options : (OrtThreadingOptions** -> OrtStatus*)
      release_threading_options : (OrtThreadingOptions* -> Void)

      model_metadata_get_custom_metadata_map_keys : (OrtModelMetadata*, OrtAllocator*, LibC::Char***, Int64* -> OrtStatus*)

      add_free_dimension_override_by_name : (OrtSessionOptions*, LibC::Char*, Int64 -> OrtStatus*)

      get_available_providers : (LibC::Char***, Int32* -> OrtStatus*)
      release_available_providers : (LibC::Char**, Int32 -> OrtStatus*)

      get_string_tensor_element_length : (OrtValue*, LibC::SizeT, LibC::SizeT* -> OrtStatus*)
      get_string_tensor_element : (OrtValue*, LibC::SizeT, LibC::SizeT, Void* -> OrtStatus*)
      fill_string_tensor_element : (OrtValue*, LibC::Char*, LibC::SizeT -> OrtStatus*)

      add_session_config_entry : (OrtSessionOptions*, LibC::Char*, LibC::Char* -> OrtStatus*)

      create_allocator : (OrtSession*, OrtMemoryInfo*, OrtAllocator** -> OrtStatus*)
      release_allocator : (OrtAllocator* -> Void)

      run_with_binding : (OrtSession*, OrtRunOptions*, OrtIoBinding* -> OrtStatus*)
      create_io_binding : (OrtSession*, OrtIoBinding** -> OrtStatus*)

      release_io_binding : (OrtIoBinding* -> Void)
      bind_input : (OrtIoBinding*, LibC::Char*, OrtValue* -> OrtStatus*)
      bind_output : (OrtIoBinding*, LibC::Char*, OrtValue* -> OrtStatus*)
      bind_output_to_device : (OrtIoBinding*, LibC::Char*, OrtMemoryInfo* -> OrtStatus*)
      get_bound_output_names : (OrtIoBinding*, OrtAllocator*, LibC::Char**, LibC::SizeT**, LibC::SizeT* -> OrtStatus*)
      get_bound_output_values : (OrtIoBinding*, OrtAllocator*, OrtValue***, LibC::SizeT* -> OrtStatus*)
      clear_bound_inputs : (OrtIoBinding* -> Void)
      clear_bound_outputs : (OrtIoBinding* -> Void)

      tensor_at : (OrtValue*, Int64*, LibC::SizeT, Void** -> OrtStatus*)

      create_and_register_allocator : (OrtEnv*, OrtMemoryInfo*, OrtArenaCfg* -> OrtStatus*)
      set_language_projection : (OrtEnv*, OrtLanguageProjection -> OrtStatus*)
      session_get_profiling_start_time_ns : (OrtSession*, UInt64* -> OrtStatus*)

      set_global_intra_op_num_threads : (OrtThreadingOptions*, Int32 -> OrtStatus*)
      set_global_inter_op_num_threads : (OrtThreadingOptions*, Int32 -> OrtStatus*)
      set_global_spin_control : (OrtThreadingOptions*, Int32 -> OrtStatus*)

      add_initializer : (OrtSessionOptions*, LibC::Char*, OrtValue* -> OrtStatus*)

      create_env_with_custom_logger_and_global_thread_pools : (OrtLoggingFunction, Void*, OrtLoggingLevel, LibC::Char*, OrtThreadingOptions*, OrtEnv** -> OrtStatus*)

      session_options_append_execution_provider_cuda : (OrtSessionOptions*, OrtCUDAProviderOptions* -> OrtStatus*)
      session_options_append_execution_provider_rocm : (OrtSessionOptions*, OrtROCMProviderOptions* -> OrtStatus*)
      session_options_append_execution_provider_open_vino : (OrtSessionOptions*, OrtOpenVINOProviderOptions* -> OrtStatus*)

      set_global_denormal_as_zero : (OrtThreadingOptions* -> OrtStatus*)

      create_arena_cfg : (LibC::SizeT, Int32, Int32, Int32, OrtArenaCfg** -> OrtStatus*)
      release_arena_cfg : (OrtArenaCfg* -> Void)

      model_metadata_get_graph_description : (OrtModelMetadata*, OrtAllocator*, LibC::Char** -> OrtStatus*)

      session_options_append_execution_provider_tensor_rt : (OrtSessionOptions*, OrtTensorRTProviderOptions* -> OrtStatus*)

      set_current_gpu_device_id : (Int32 -> OrtStatus*)
      get_current_gpu_device_id : (Int32* -> OrtStatus*)

      kernel_info_get_attribute_array_float : (OrtKernelInfo*, LibC::Char*, Float32*, LibC::SizeT* -> OrtStatus*)
      kernel_info_get_attribute_array_int64 : (OrtKernelInfo*, LibC::Char*, Int64*, LibC::SizeT* -> OrtStatus*)

      create_arena_cfg_v2 : (LibC::Char**, LibC::SizeT*, LibC::SizeT, OrtArenaCfg** -> OrtStatus*)

      add_run_config_entry : (OrtRunOptions*, LibC::Char*, LibC::Char* -> OrtStatus*)

      create_prepacked_weights_container : (OrtPrepackedWeightsContainer** -> OrtStatus*)
      release_prepacked_weights_container : (OrtPrepackedWeightsContainer* -> Void)

      create_session_with_prepacked_weights_container : (OrtEnv*, ORTCHAR_T*, OrtSessionOptions*, OrtPrepackedWeightsContainer*, OrtSession** -> OrtStatus*)
      create_session_from_array_with_prepacked_weights_container : (OrtEnv*, Void*, LibC::SizeT, OrtSessionOptions*, OrtPrepackedWeightsContainer*, OrtSession** -> OrtStatus*)

      session_options_append_execution_provider_tensor_rt_v2 : (OrtSessionOptions*, OrtTensorRTProviderOptionsV2* -> OrtStatus*)
      create_tensor_rt_provider_options : (OrtTensorRTProviderOptionsV2** -> OrtStatus*)
      update_tensor_rt_provider_options : (OrtTensorRTProviderOptionsV2*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      get_tensor_rt_provider_options_as_string : (OrtTensorRTProviderOptionsV2*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      release_tensor_rt_provider_options : (OrtTensorRTProviderOptionsV2* -> Void)

      enable_ort_custom_ops : (OrtSessionOptions* -> OrtStatus*)

      register_allocator : (OrtEnv*, OrtAllocator* -> OrtStatus*)
      unregister_allocator : (OrtEnv*, OrtMemoryInfo* -> OrtStatus*)

      is_sparse_tensor : (OrtValue*, Int32* -> OrtStatus*)
      create_sparse_tensor_as_ort_value : (OrtAllocator*, Int64*, LibC::SizeT, ONNXTensorElementDataType, OrtValue** -> OrtStatus*)
      fill_sparse_tensor_coo : (OrtValue*, OrtMemoryInfo*, Int64*, LibC::SizeT, Void*, Int64*, LibC::SizeT -> OrtStatus*)
      fill_sparse_tensor_csr : (OrtValue*, OrtMemoryInfo*, Int64*, LibC::SizeT, Void*, Int64*, LibC::SizeT, Int64*, LibC::SizeT -> OrtStatus*)
      fill_sparse_tensor_block_sparse : (OrtValue*, OrtMemoryInfo*, Int64*, LibC::SizeT, Void*, Int64*, LibC::SizeT, Int32* -> OrtStatus*)
      create_sparse_tensor_with_values_as_ort_value : (OrtMemoryInfo*, Void*, Int64*, LibC::SizeT, Int64*, LibC::SizeT, ONNXTensorElementDataType, OrtValue** -> OrtStatus*)
      use_coo_indices : (OrtValue*, Int64*, LibC::SizeT -> OrtStatus*)
      use_csr_indices : (OrtValue*, Int64*, LibC::SizeT, Int64*, LibC::SizeT -> OrtStatus*)
      use_block_sparse_indices : (OrtValue*, Int64*, LibC::SizeT, Int32* -> OrtStatus*)
      get_sparse_tensor_format : (OrtValue*, OrtSparseFormat* -> OrtStatus*)
      get_sparse_tensor_values_type_and_shape : (OrtValue*, OrtTensorTypeAndShapeInfo** -> OrtStatus*)
      get_sparse_tensor_values : (OrtValue*, Void** -> OrtStatus*)
      get_sparse_tensor_indices_type_shape : (OrtValue*, OrtSparseIndicesFormat, OrtTensorTypeAndShapeInfo** -> OrtStatus*)
      get_sparse_tensor_indices : (OrtValue*, OrtSparseIndicesFormat, LibC::SizeT*, Void** -> OrtStatus*)

      has_value : (OrtValue*, Int32* -> OrtStatus*)

      kernel_context_get_gpu_compute_stream : (OrtKernelContext*, Void** -> OrtStatus*)

      get_tensor_memory_info : (OrtValue*, OrtMemoryInfo** -> OrtStatus*)

      get_execution_provider_api : (LibC::Char*, UInt32, Void** -> OrtStatus*)

      session_options_set_custom_create_thread_fn : (OrtSessionOptions*, OrtCustomCreateThreadFn -> OrtStatus*)
      session_options_set_custom_thread_creation_options : (OrtSessionOptions*, Void* -> OrtStatus*)
      session_options_set_custom_join_thread_fn : (OrtSessionOptions*, OrtCustomJoinThreadFn -> OrtStatus*)

      set_global_custom_create_thread_fn : (OrtThreadingOptions*, OrtCustomCreateThreadFn -> OrtStatus*)
      set_global_custom_thread_creation_options : (OrtThreadingOptions*, Void* -> OrtStatus*)
      set_global_custom_join_thread_fn : (OrtThreadingOptions*, OrtCustomJoinThreadFn -> OrtStatus*)

      synchronize_bound_inputs : (OrtIoBinding* -> OrtStatus*)
      synchronize_bound_outputs : (OrtIoBinding* -> OrtStatus*)

      session_options_append_execution_provider_cuda_v2 : (OrtSessionOptions*, OrtCUDAProviderOptionsV2* -> OrtStatus*)
      create_cuda_provider_options : (OrtCUDAProviderOptionsV2** -> OrtStatus*)
      update_cuda_provider_options : (OrtCUDAProviderOptionsV2*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      get_cuda_provider_options_as_string : (OrtCUDAProviderOptionsV2*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      release_cuda_provider_options : (OrtCUDAProviderOptionsV2* -> Void)

      session_options_append_execution_provider_mi_graph_x : (OrtSessionOptions*, OrtMIGraphXProviderOptions* -> OrtStatus*)

      add_external_initializers : (OrtSessionOptions*, LibC::Char**, OrtValue**, LibC::SizeT -> OrtStatus*)
      create_op_attr : (LibC::Char*, Void*, Int32, OrtOpAttrType, OrtOpAttr** -> OrtStatus*)
      release_op_attr : (OrtOpAttr* -> Void)
      create_op : (OrtKernelInfo*, LibC::Char*, LibC::Char*, Int32, LibC::Char**, ONNXTensorElementDataType*, Int32, OrtOpAttr**, Int32, Int32, Int32, OrtOp** -> OrtStatus*)
      invoke_op : (OrtKernelContext*, OrtOp*, OrtValue**, Int32, OrtValue**, Int32 -> OrtStatus*)
      release_op : (OrtOp* -> Void)

      session_options_append_execution_provider : (OrtSessionOptions*, LibC::Char*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      copy_kernel_info : (OrtKernelInfo*, OrtKernelInfo** -> OrtStatus*)
      release_kernel_info : (OrtKernelInfo* -> Void)
      get_training_api : (UInt32 -> OrtTrainingApi*)

      session_options_append_execution_provider_cann : (OrtSessionOptions*, OrtCANNProviderOptions* -> OrtStatus*)
      create_cann_provider_options : (OrtCANNProviderOptions** -> OrtStatus*)
      update_cann_provider_options : (OrtCANNProviderOptions*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      get_cann_provider_options_as_string : (OrtCANNProviderOptions*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      release_cann_provider_options : (OrtCANNProviderOptions* -> Void)

      memory_info_get_device_type : (OrtMemoryInfo*, OrtMemoryInfoDeviceType* -> Void)
      update_env_with_custom_log_level : (OrtEnv*, OrtLoggingLevel -> OrtStatus*)
      set_global_intra_op_thread_affinity : (OrtThreadingOptions*, LibC::Char* -> OrtStatus*)
      register_custom_ops_library_v2 : (OrtSessionOptions*, ORTCHAR_T* -> OrtStatus*)
      register_custom_ops_using_function : (OrtSessionOptions*, LibC::Char* -> OrtStatus*)

      kernel_info_get_input_count : (OrtKernelInfo*, LibC::SizeT* -> OrtStatus*)
      kernel_info_get_output_count : (OrtKernelInfo*, LibC::SizeT* -> OrtStatus*)
      kernel_info_get_input_name : (OrtKernelInfo*, LibC::SizeT, LibC::Char*, LibC::SizeT* -> OrtStatus*)
      kernel_info_get_output_name : (OrtKernelInfo*, LibC::SizeT, LibC::Char*, LibC::SizeT* -> OrtStatus*)
      kernel_info_get_input_type_info : (OrtKernelInfo*, LibC::SizeT, OrtTypeInfo** -> OrtStatus*)
      kernel_info_get_output_type_info : (OrtKernelInfo*, LibC::SizeT, OrtTypeInfo** -> OrtStatus*)
      kernel_info_get_attribute_tensor : (OrtKernelInfo*, LibC::Char*, OrtAllocator*, OrtValue** -> OrtStatus*)

      has_session_config_entry : (OrtSessionOptions*, LibC::Char*, Int32* -> OrtStatus*)
      get_session_config_entry : (OrtSessionOptions*, LibC::Char*, LibC::Char*, LibC::SizeT* -> OrtStatus*)

      session_options_append_execution_provider_dnnl : (OrtSessionOptions*, OrtDnnlProviderOptions* -> OrtStatus*)
      create_dnnl_provider_options : (OrtDnnlProviderOptions** -> OrtStatus*)
      update_dnnl_provider_options : (OrtDnnlProviderOptions*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      get_dnnl_provider_options_as_string : (OrtDnnlProviderOptions*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      release_dnnl_provider_options : (OrtDnnlProviderOptions* -> Void)

      kernel_info_get_node_name : (OrtKernelInfo*, LibC::Char*, LibC::SizeT* -> OrtStatus*)
      kernel_info_get_logger : (OrtKernelInfo*, OrtLogger** -> OrtStatus*)
      kernel_context_get_logger : (OrtKernelContext*, OrtLogger** -> OrtStatus*)
      logger_log_message : (OrtLogger*, OrtLoggingLevel, LibC::Char*, ORTCHAR_T*, Int32, LibC::Char* -> OrtStatus*)
      logger_get_logging_severity_level : (OrtLogger*, OrtLoggingLevel* -> OrtStatus*)

      kernel_info_get_constant_input_tensor : (OrtKernelInfo*, LibC::SizeT, Int32*, OrtValue** -> OrtStatus*)
      cast_type_info_to_optional_type_info : (OrtTypeInfo*, OrtOptionalTypeInfo** -> OrtStatus*)
      get_optional_contained_type_info : (OrtOptionalTypeInfo*, OrtTypeInfo** -> OrtStatus*)
      get_resized_string_tensor_element_buffer : (OrtValue*, LibC::SizeT, LibC::SizeT, LibC::Char** -> OrtStatus*)
      kernel_context_get_allocator : (OrtKernelContext*, OrtMemoryInfo*, OrtAllocator** -> OrtStatus*)
      get_build_info_string : (-> LibC::Char*)

      create_rocm_provider_options : (OrtROCMProviderOptions** -> OrtStatus*)
      update_rocm_provider_options : (OrtROCMProviderOptions*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      get_rocm_provider_options_as_string : (OrtROCMProviderOptions*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      release_rocm_provider_options : (OrtROCMProviderOptions* -> Void)

      create_and_register_allocator_v2 : (OrtEnv*, LibC::Char*, OrtMemoryInfo*, OrtArenaCfg*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      run_async : (OrtSession*, OrtRunOptions*, LibC::Char**, OrtValue**, LibC::SizeT, LibC::Char**, LibC::SizeT, OrtValue**, RunAsyncCallbackFn, Void* -> OrtStatus*)
      update_tensor_rt_provider_options_with_value : (OrtTensorRTProviderOptionsV2*, LibC::Char*, Void* -> OrtStatus*)
      get_tensor_rt_provider_options_by_name : (OrtTensorRTProviderOptionsV2*, LibC::Char*, Void** -> OrtStatus*)
      update_cuda_provider_options_with_value : (OrtCUDAProviderOptionsV2*, LibC::Char*, Void* -> OrtStatus*)
      get_cuda_provider_options_by_name : (OrtCUDAProviderOptionsV2*, LibC::Char*, Void** -> OrtStatus*)
      kernel_context_get_resource : (OrtKernelContext*, Int32, Int32, Void** -> OrtStatus*)

      set_user_logging_function : (OrtSessionOptions*, OrtLoggingFunction, Void* -> OrtStatus*)

      shape_infer_context_get_input_count : (OrtShapeInferContext*, LibC::SizeT* -> OrtStatus*)
      shape_infer_context_get_input_type_shape : (OrtShapeInferContext*, LibC::SizeT, OrtTensorTypeAndShapeInfo** -> OrtStatus*)
      shape_infer_context_get_attribute : (OrtShapeInferContext*, LibC::Char*, OrtOpAttr** -> OrtStatus*)
      shape_infer_context_set_output_type_shape : (OrtShapeInferContext*, LibC::SizeT, OrtTensorTypeAndShapeInfo* -> OrtStatus*)
      set_symbolic_dimensions : (OrtTensorTypeAndShapeInfo*, LibC::Char**, LibC::SizeT -> OrtStatus*)
      read_op_attr : (OrtOpAttr*, OrtOpAttrType, Void*, LibC::SizeT, LibC::SizeT* -> OrtStatus*)
      set_deterministic_compute : (OrtSessionOptions*, Bool -> OrtStatus*)
      kernel_context_parallel_for : (OrtKernelContext*, (Void*, LibC::SizeT -> Void), LibC::SizeT, LibC::SizeT, Void* -> OrtStatus*)

      session_options_append_execution_provider_open_vino_v2 : (OrtSessionOptions*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      session_options_append_execution_provider_vitis_ai : (OrtSessionOptions*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)

      kernel_context_get_scratch_buffer : (OrtKernelContext*, OrtMemoryInfo*, LibC::SizeT, Void** -> OrtStatus*)
      kernel_info_get_allocator : (OrtKernelInfo*, OrtMemType, OrtAllocator** -> OrtStatus*)
      add_external_initializers_from_files_in_memory : (OrtSessionOptions*, ORTCHAR_T**, LibC::Char**, LibC::SizeT*, LibC::SizeT -> OrtStatus*)

      create_lora_adapter : (ORTCHAR_T*, OrtAllocator*, OrtLoraAdapter** -> OrtStatus*)
      create_lora_adapter_from_array : (Void*, LibC::SizeT, OrtAllocator*, OrtLoraAdapter** -> OrtStatus*)
      release_lora_adapter : (OrtLoraAdapter* -> Void)
      run_options_add_active_lora_adapter : (OrtRunOptions*, OrtLoraAdapter* -> OrtStatus*)

      set_ep_dynamic_options : (OrtSession*, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)

      release_value_info : (OrtValueInfo* -> Void)
      release_node : (OrtNode* -> Void)
      release_graph : (OrtGraph* -> Void)
      release_model : (OrtModel* -> Void)

      get_value_info_name : (OrtValueInfo*, LibC::Char** -> OrtStatus*)
      get_value_info_type_info : (OrtValueInfo*, OrtTypeInfo** -> OrtStatus*)

      get_model_editor_api : (-> OrtModelEditorApi*)
      create_tensor_with_data_and_deleter_as_ort_value : (OrtAllocator*, Void*, LibC::SizeT, Int64*, LibC::SizeT, ONNXTensorElementDataType, OrtValue** -> OrtStatus*)
      session_options_set_load_cancellation_flag : (OrtSessionOptions*, Bool -> OrtStatus*)
      get_compile_api : (-> OrtCompileApi*)

      create_key_value_pairs : (OrtKeyValuePairs** -> Void)
      add_key_value_pair : (OrtKeyValuePairs*, LibC::Char*, LibC::Char* -> Void)
      get_key_value : (OrtKeyValuePairs*, LibC::Char* -> LibC::Char*)
      get_key_value_pairs : (OrtKeyValuePairs*, LibC::Char***, LibC::Char***, LibC::SizeT* -> Void)
      remove_key_value_pair : (OrtKeyValuePairs*, LibC::Char* -> Void)
      release_key_value_pairs : (OrtKeyValuePairs* -> Void)

      register_execution_provider_library : (OrtEnv*, LibC::Char*, ORTCHAR_T* -> OrtStatus*)
      unregister_execution_provider_library : (OrtEnv*, LibC::Char* -> OrtStatus*)
      get_ep_devices : (OrtEnv*, OrtEpDevice***, LibC::SizeT* -> OrtStatus*)

      session_options_append_execution_provider_v2 : (OrtSessionOptions*, OrtEnv*, OrtEpDevice**, LibC::SizeT, LibC::Char**, LibC::Char**, LibC::SizeT -> OrtStatus*)
      session_options_set_ep_selection_policy : (OrtSessionOptions*, OrtExecutionProviderDevicePolicy -> OrtStatus*)
      session_options_set_ep_selection_policy_delegate : (OrtSessionOptions*, EpSelectionDelegate, Void* -> OrtStatus*)

      hardware_device_type : (OrtHardwareDevice* -> OrtHardwareDeviceType)
      hardware_device_vendor_id : (OrtHardwareDevice* -> UInt32)
      hardware_device_vendor : (OrtHardwareDevice* -> LibC::Char*)
      hardware_device_device_id : (OrtHardwareDevice* -> UInt32)
      hardware_device_metadata : (OrtHardwareDevice* -> OrtKeyValuePairs*)

      ep_device_ep_name : (OrtEpDevice* -> LibC::Char*)
      ep_device_ep_vendor : (OrtEpDevice* -> LibC::Char*)
      ep_device_ep_metadata : (OrtEpDevice* -> OrtKeyValuePairs*)
      ep_device_ep_options : (OrtEpDevice* -> OrtKeyValuePairs*)
      ep_device_device : (OrtEpDevice* -> OrtHardwareDevice*)
      get_ep_api : (-> OrtEpApi*)

      get_tensor_size_in_bytes : (OrtValue*, LibC::SizeT* -> OrtStatus*)
      allocator_get_stats : (OrtAllocator*, OrtKeyValuePairs** -> OrtStatus*)

      create_memory_info_v2 : (LibC::Char*, OrtMemoryInfoDeviceType, UInt32, Int32, OrtDeviceMemoryType, LibC::SizeT, OrtAllocatorType, OrtMemoryInfo** -> OrtStatus*)
      memory_info_get_device_mem_type : (OrtMemoryInfo* -> OrtDeviceMemoryType)
      memory_info_get_vendor_id : (OrtMemoryInfo* -> UInt32)

      value_info_get_value_producer : (OrtValueInfo*, OrtNode**, LibC::SizeT* -> OrtStatus*)
      value_info_get_value_num_consumers : (OrtValueInfo*, LibC::SizeT* -> OrtStatus*)
      value_info_get_value_consumers : (OrtValueInfo*, OrtNode**, Int64*, LibC::SizeT -> OrtStatus*)
      value_info_get_initializer_value : (OrtValueInfo*, OrtValue** -> OrtStatus*)
      value_info_get_external_initializer_info : (OrtValueInfo*, OrtExternalInitializerInfo** -> OrtStatus*)
      value_info_is_required_graph_input : (OrtValueInfo*, Bool* -> OrtStatus*)
      value_info_is_optional_graph_input : (OrtValueInfo*, Bool* -> OrtStatus*)
      value_info_is_graph_output : (OrtValueInfo*, Bool* -> OrtStatus*)
      value_info_is_constant_initializer : (OrtValueInfo*, Bool* -> OrtStatus*)
      value_info_is_from_outer_scope : (OrtValueInfo*, Bool* -> OrtStatus*)

      graph_get_name : (OrtGraph*, LibC::Char** -> OrtStatus*)
      graph_get_model_path : (OrtGraph*, ORTCHAR_T** -> OrtStatus*)
      graph_get_onnx_ir_version : (OrtGraph*, Int64* -> OrtStatus*)
      graph_get_num_operator_sets : (OrtGraph*, LibC::SizeT* -> OrtStatus*)
      graph_get_operator_sets : (OrtGraph*, LibC::Char**, Int64*, LibC::SizeT -> OrtStatus*)
      graph_get_num_inputs : (OrtGraph*, LibC::SizeT* -> OrtStatus*)
      graph_get_inputs : (OrtGraph*, OrtValueInfo**, LibC::SizeT -> OrtStatus*)
      graph_get_num_outputs : (OrtGraph*, LibC::SizeT* -> OrtStatus*)
      graph_get_outputs : (OrtGraph*, OrtValueInfo**, LibC::SizeT -> OrtStatus*)
      graph_get_num_initializers : (OrtGraph*, LibC::SizeT* -> OrtStatus*)
      graph_get_initializers : (OrtGraph*, OrtValueInfo**, LibC::SizeT -> OrtStatus*)
      graph_get_num_nodes : (OrtGraph*, LibC::SizeT* -> OrtStatus*)
      graph_get_nodes : (OrtGraph*, OrtNode**, LibC::SizeT -> OrtStatus*)
      graph_get_parent_node : (OrtGraph*, OrtNode** -> OrtStatus*)
      graph_get_graph_view : (OrtGraph*, OrtNode**, LibC::SizeT, OrtGraph** -> OrtStatus*)

      node_get_id : (OrtNode*, LibC::SizeT* -> OrtStatus*)
      node_get_name : (OrtNode*, LibC::Char** -> OrtStatus*)
      node_get_operator_type : (OrtNode*, LibC::Char** -> OrtStatus*)
      node_get_domain : (OrtNode*, LibC::Char** -> OrtStatus*)
      node_get_since_version : (OrtNode*, Int32* -> OrtStatus*)
      node_get_num_inputs : (OrtNode*, LibC::SizeT* -> OrtStatus*)
      node_get_inputs : (OrtNode*, OrtValueInfo**, LibC::SizeT -> OrtStatus*)
      node_get_num_outputs : (OrtNode*, LibC::SizeT* -> OrtStatus*)
      node_get_outputs : (OrtNode*, OrtValueInfo**, LibC::SizeT -> OrtStatus*)
      node_get_num_implicit_inputs : (OrtNode*, LibC::SizeT* -> OrtStatus*)
      node_get_implicit_inputs : (OrtNode*, OrtValueInfo**, LibC::SizeT -> OrtStatus*)
      node_get_num_attributes : (OrtNode*, LibC::SizeT* -> OrtStatus*)
      node_get_attributes : (OrtNode*, OrtOpAttr**, LibC::SizeT -> OrtStatus*)
      node_get_attribute_by_name : (OrtNode*, LibC::Char*, OrtOpAttr** -> OrtStatus*)
      op_attr_get_tensor_attribute_as_ort_value : (OrtOpAttr*, OrtValue** -> OrtStatus*)
      op_attr_get_type : (OrtOpAttr*, OrtOpAttrType* -> OrtStatus*)
      op_attr_get_name : (OrtOpAttr*, LibC::Char** -> OrtStatus*)
      node_get_num_subgraphs : (OrtNode*, LibC::SizeT* -> OrtStatus*)
      node_get_subgraphs : (OrtNode*, OrtGraph**, LibC::SizeT, LibC::Char** -> OrtStatus*)
      node_get_graph : (OrtNode*, OrtGraph** -> OrtStatus*)
      node_get_ep_name : (OrtNode*, LibC::Char** -> OrtStatus*)

      release_external_initializer_info : (OrtExternalInitializerInfo* -> Void)
      external_initializer_info_get_file_path : (OrtExternalInitializerInfo* -> ORTCHAR_T*)
      external_initializer_info_get_file_offset : (OrtExternalInitializerInfo* -> Int64)
      external_initializer_info_get_byte_size : (OrtExternalInitializerInfo* -> LibC::SizeT)

      get_run_config_entry : (OrtRunOptions*, LibC::Char* -> LibC::Char*)

      ep_device_memory_info : (OrtEpDevice*, OrtDeviceMemoryType -> OrtMemoryInfo*)
      create_shared_allocator : (OrtEnv*, OrtEpDevice*, OrtDeviceMemoryType, OrtAllocatorType, OrtKeyValuePairs*, OrtAllocator** -> OrtStatus*)
      get_shared_allocator : (OrtEnv*, OrtMemoryInfo*, OrtAllocator** -> OrtStatus*)
      release_shared_allocator : (OrtEnv*, OrtEpDevice*, OrtDeviceMemoryType -> OrtStatus*)

      get_tensor_data : (OrtValue*, Void** -> OrtStatus*)

      get_session_options_config_entries : (OrtSessionOptions*, OrtKeyValuePairs** -> OrtStatus*)

      session_get_memory_info_for_inputs : (OrtSession*, OrtMemoryInfo**, LibC::SizeT -> OrtStatus*)
      session_get_memory_info_for_outputs : (OrtSession*, OrtMemoryInfo**, LibC::SizeT -> OrtStatus*)
      session_get_ep_device_for_inputs : (OrtSession*, OrtEpDevice**, LibC::SizeT -> OrtStatus*)
      create_sync_stream_for_ep_device : (OrtEpDevice*, OrtKeyValuePairs*, OrtSyncStream** -> OrtStatus*)
      sync_stream_get_handle : (OrtSyncStream* -> Void*)
      release_sync_stream : (OrtSyncStream* -> Void)
      copy_tensors : (OrtEnv*, OrtValue**, OrtValue**, OrtSyncStream*, LibC::SizeT -> OrtStatus*)
      graph_get_model_metadata : (OrtGraph*, OrtModelMetadata** -> OrtStatus*)
      get_model_compatibility_for_ep_devices : (OrtEpDevice**, LibC::SizeT, LibC::Char*, OrtCompiledModelCompatibility* -> OrtStatus*)
      create_external_initializer_info : (ORTCHAR_T*, Int64, LibC::SizeT, OrtExternalInitializerInfo** -> OrtStatus*)
      tensor_type_and_shape_has_shape : (OrtTensorTypeAndShapeInfo* -> Bool)
      kernel_info_get_config_entries : (OrtKernelInfo*, OrtKeyValuePairs** -> OrtStatus*)
      kernel_info_get_operator_domain : (OrtKernelInfo*, LibC::Char*, LibC::SizeT* -> OrtStatus*)
      kernel_info_get_operator_type : (OrtKernelInfo*, LibC::Char*, LibC::SizeT* -> OrtStatus*)
      kernel_info_get_operator_since_version : (OrtKernelInfo*, Int32* -> OrtStatus*)
      get_interop_api : (-> OrtInteropApi*)
      session_get_ep_device_for_outputs : (OrtSession*, OrtEpDevice**, LibC::SizeT -> OrtStatus*)
      get_num_hardware_devices : (OrtEnv*, LibC::SizeT* -> OrtStatus*)
      get_hardware_devices : (OrtEnv*, OrtHardwareDevice**, LibC::SizeT -> OrtStatus*)
      get_hardware_device_ep_incompatibility_details : (OrtEnv*, LibC::Char*, OrtHardwareDevice*, OrtDeviceEpIncompatibilityDetails** -> OrtStatus*)
      device_ep_incompatibility_details_get_reasons_bitmask : (OrtDeviceEpIncompatibilityDetails*, UInt32* -> OrtStatus*)
      device_ep_incompatibility_details_get_notes : (OrtDeviceEpIncompatibilityDetails*, LibC::Char** -> OrtStatus*)
      device_ep_incompatibility_details_get_error_code : (OrtDeviceEpIncompatibilityDetails*, Int32* -> OrtStatus*)
      release_device_ep_incompatibility_details : (OrtDeviceEpIncompatibilityDetails* -> Void)
      get_compatibility_info_from_model : (ORTCHAR_T*, LibC::Char*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      get_compatibility_info_from_model_bytes : (Void*, LibC::SizeT, LibC::Char*, OrtAllocator*, LibC::Char** -> OrtStatus*)
      create_env_with_options : (EnvCreationOptions*, OrtEnv** -> OrtStatus*)
      session_get_ep_graph_assignment_info : (OrtSession*, OrtEpAssignedSubgraph***, LibC::SizeT* -> OrtStatus*)
      ep_assigned_subgraph_get_ep_name : (OrtEpAssignedSubgraph*, LibC::Char** -> OrtStatus*)
      ep_assigned_subgraph_get_nodes : (OrtEpAssignedSubgraph*, OrtEpAssignedNode***, LibC::SizeT* -> OrtStatus*)
      ep_assigned_node_get_name : (OrtEpAssignedNode*, LibC::Char** -> OrtStatus*)
      ep_assigned_node_get_domain : (OrtEpAssignedNode*, LibC::Char** -> OrtStatus*)
      ep_assigned_node_get_operator_type : (OrtEpAssignedNode*, LibC::Char** -> OrtStatus*)
      run_options_set_sync_stream : (OrtRunOptions*, OrtSyncStream* -> Void)
      get_tensor_element_type_and_shape_data_reference : (OrtValue*, ONNXTensorElementDataType*, Int64**, LibC::SizeT* -> OrtStatus*)
    end

    struct OrtModelEditorApi
      create_tensor_type_info : (OrtTensorTypeAndShapeInfo*, OrtTypeInfo** -> OrtStatus*)
      create_sparse_tensor_type_info : (OrtTensorTypeAndShapeInfo*, OrtTypeInfo** -> OrtStatus*)
      create_map_type_info : (ONNXTensorElementDataType, OrtTypeInfo*, OrtTypeInfo** -> OrtStatus*)
      create_sequence_type_info : (OrtTypeInfo*, OrtTypeInfo** -> OrtStatus*)
      create_optional_type_info : (OrtTypeInfo*, OrtTypeInfo** -> OrtStatus*)
      create_value_info : (LibC::Char*, OrtTypeInfo*, OrtValueInfo** -> OrtStatus*)
      create_node : (LibC::Char*, LibC::Char*, LibC::Char*, LibC::Char**, LibC::SizeT, LibC::Char**, LibC::SizeT, OrtOpAttr**, LibC::SizeT, OrtNode** -> OrtStatus*)
      create_graph : (OrtGraph** -> OrtStatus*)
      set_graph_inputs : (OrtGraph*, OrtValueInfo**, LibC::SizeT -> OrtStatus*)
      set_graph_outputs : (OrtGraph*, OrtValueInfo**, LibC::SizeT -> OrtStatus*)
      add_initializer_to_graph : (OrtGraph*, LibC::Char*, OrtValue*, Bool -> OrtStatus*)
      add_node_to_graph : (OrtGraph*, OrtNode* -> OrtStatus*)
      create_model : (LibC::Char**, Int32*, LibC::SizeT, OrtModel** -> OrtStatus*)
      add_graph_to_model : (OrtModel*, OrtGraph* -> OrtStatus*)
      create_session_from_model : (OrtEnv*, OrtModel*, OrtSessionOptions*, OrtSession** -> OrtStatus*)
      create_model_editor_session : (OrtEnv*, ORTCHAR_T*, OrtSessionOptions*, OrtSession** -> OrtStatus*)
      create_model_editor_session_from_array : (OrtEnv*, Void*, LibC::SizeT, OrtSessionOptions*, OrtSession** -> OrtStatus*)
      session_get_opset_for_domain : (OrtSession*, LibC::Char*, Int32* -> OrtStatus*)
      apply_model_to_model_editor_session : (OrtSession*, OrtModel* -> OrtStatus*)
      finalize_model_editor_session : (OrtSession*, OrtSessionOptions*, OrtPrepackedWeightsContainer* -> OrtStatus*)
    end

    struct OrtCompileApi
      release_model_compilation_options : (OrtModelCompilationOptions* -> Void)
      create_model_compilation_options_from_session_options : (OrtEnv*, OrtSessionOptions*, OrtModelCompilationOptions** -> OrtStatus*)
      model_compilation_options_set_input_model_path : (OrtModelCompilationOptions*, ORTCHAR_T* -> OrtStatus*)
      model_compilation_options_set_input_model_from_buffer : (OrtModelCompilationOptions*, Void*, LibC::SizeT -> OrtStatus*)
      model_compilation_options_set_output_model_path : (OrtModelCompilationOptions*, ORTCHAR_T* -> OrtStatus*)
      model_compilation_options_set_output_model_external_initializers_file : (OrtModelCompilationOptions*, ORTCHAR_T*, LibC::SizeT -> OrtStatus*)
      model_compilation_options_set_output_model_buffer : (OrtModelCompilationOptions*, OrtAllocator*, Void**, LibC::SizeT* -> OrtStatus*)
      model_compilation_options_set_ep_context_embed_mode : (OrtModelCompilationOptions*, Bool -> OrtStatus*)
      compile_model : (OrtEnv*, OrtModelCompilationOptions* -> OrtStatus*)
      model_compilation_options_set_flags : (OrtModelCompilationOptions*, UInt32 -> OrtStatus*)
      model_compilation_options_set_ep_context_binary_information : (OrtModelCompilationOptions*, ORTCHAR_T*, ORTCHAR_T* -> OrtStatus*)
      model_compilation_options_set_graph_optimization_level : (OrtModelCompilationOptions*, GraphOptimizationLevel -> OrtStatus*)
      model_compilation_options_set_output_model_write_func : (OrtModelCompilationOptions*, OrtWriteBufferFunc, Void* -> OrtStatus*)
      model_compilation_options_set_output_model_get_initializer_location_func : (OrtModelCompilationOptions*, OrtGetInitializerLocationFunc, Void* -> OrtStatus*)
    end

    struct OrtInteropApi
      create_external_resource_importer_for_device : (OrtEpDevice*, OrtExternalResourceImporter** -> OrtStatus*)
      release_external_resource_importer : (OrtExternalResourceImporter* -> Void)
      can_import_memory : (OrtExternalResourceImporter*, OrtExternalMemoryHandleType, Bool* -> OrtStatus*)
      import_memory : (OrtExternalResourceImporter*, OrtExternalMemoryDescriptor*, OrtExternalMemoryHandle** -> OrtStatus*)
      release_external_memory_handle : (OrtExternalMemoryHandle* -> Void)
      create_tensor_from_memory : (OrtExternalResourceImporter*, OrtExternalMemoryHandle*, OrtExternalTensorDescriptor*, OrtValue** -> OrtStatus*)
      can_import_semaphore : (OrtExternalResourceImporter*, OrtExternalSemaphoreType, Bool* -> OrtStatus*)
      import_semaphore : (OrtExternalResourceImporter*, OrtExternalSemaphoreDescriptor*, OrtExternalSemaphoreHandle** -> OrtStatus*)
      release_external_semaphore_handle : (OrtExternalSemaphoreHandle* -> Void)
      wait_semaphore : (OrtExternalResourceImporter*, OrtExternalSemaphoreHandle*, OrtSyncStream*, UInt64 -> OrtStatus*)
      signal_semaphore : (OrtExternalResourceImporter*, OrtExternalSemaphoreHandle*, OrtSyncStream*, UInt64 -> OrtStatus*)
    end

    struct ApiBase
      get_api : (UInt32 -> Api*)
      get_version_string : (-> LibC::Char*)
    end

    fun OrtGetApiBase : ApiBase*
  end
end
