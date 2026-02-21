# onnxruntime.cr Development Guidelines

## Memory Management Overview

This library provides Crystal bindings to the ONNX Runtime C API. Proper memory management is critical to avoid resource leaks, double-free errors, and use-after-free bugs when working with C resources from Crystal.

### Key Principles

- **Explicit Resource Allocation**: For each ORT structure, decide whether Crystal or C is responsible for allocation
- **Avoid Mixed Ownership**: Never allow both Crystal and C to manage the same resource—this leads to double-free errors
- **Track Release State**: Use `@resource_released` flags to prevent releasing the same resource twice
- **Exception Safety**: Use `begin/ensure` blocks to guarantee cleanup even when exceptions occur

### Recommended API Styles

- **RAII block style (recommended for short-lived workloads)**:
  ```crystal
  OnnxRuntime::InferenceSession.open("model.onnx", release_env: true) do |session|
    result = session.run(input_data)
  end
  ```
- **Explicit lifecycle style (recommended for long-running services)**:
  ```crystal
  session = OnnxRuntime::InferenceSession.new("model.onnx")
  result = session.run(input_data)
  session.release
  OnnxRuntime::InferenceSession.release_env
  ```

## Environment Management

ONNX Runtime requires a global environment (`OrtEnv`) that must outlive all sessions. This library manages it using a singleton pattern.

### OrtEnvironment Singleton

- **Pattern**: `OrtEnvironment` class implements thread-safe singleton pattern
- **Thread Safety**: Mutex protects concurrent access to shared environment
- **Explicit Release**: Environment must be released manually, not via finalizer
- **Usage Pattern**:
  ```crystal
  # Create multiple sessions (environment created automatically on first use)
  session1 = OnnxRuntime::InferenceSession.new("model1.onnx")
  session2 = OnnxRuntime::InferenceSession.new("model2.onnx")
  
  # Use sessions for inference...
  
  # On shutdown: release sessions first, then environment
  session1.release
  session2.release
  
  # Release shared environment last
  OnnxRuntime::InferenceSession.release_env
  ```

- **RAII Option**: `InferenceSession.open(..., release_env: true)` can release environment automatically at block exit.

### Why Manual Release?

Previous attempts to use Crystal's `finalize` for automatic cleanup caused crashes on macOS:

```
libc++abi: terminating due to uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument
```

This appears to be caused by undefined finalization order in multi-threaded contexts. Manual resource management via explicit `release_*` methods avoids this issue.

## Resource Management by Type

### OrtEnv (Global Environment)

- **Creation**: C API via `api.create_env`
- **Release**: `api.release_env` (called by `OrtEnvironment.release`)
- **Ownership**: Managed by `OrtEnvironment` singleton
- **Lifetime**: Must outlive all sessions

### OrtSession (Inference Session)

- **Creation**: C API via `api.create_session` or `api.create_session_from_array`
- **Release**: `api.release_session` (called by `InferenceSession#release` / `InferenceSession#release_session`)
- **RAII Helper**: `InferenceSession.open` guarantees `release_session` in an `ensure` block
- **Tracking**: `@session_released` flag prevents double-free
- **Lifetime**: Created per model, released before environment

### OrtValue (Tensors)

- **Creation**: C API functions like `api.create_tensor_with_data_as_ort_value`
- **Release**: `api.release_value`
- **Pattern**: Always released in `ensure` block immediately after use
- **Examples**:
  - Input tensors: Released after `session.run` completes
  - Output tensors: Released after data extraction
  - Intermediate tensors: Released in `ensure` block

### OrtAllocator (Memory Allocator)

- **Creation**: `api.get_allocator_with_default_options` (default allocator)
- **Release**: Default allocator typically doesn't need explicit release (commented out in code)
- **Usage**: Used for allocating ORT-managed memory

### OrtSessionOptions

- **Creation**: `api.create_session_options`
- **Release**: `api.release_session_options`
- **Pattern**: Released in `ensure` block after session creation

### OrtRunOptions

- **Creation**: `api.create_run_options`
- **Release**: `api.release_run_options`
- **Pattern**: Released in `ensure` block after inference

### OrtModelMetadata

- **Creation**: `api.session_get_model_metadata`
- **Release**: `api.release_model_metadata`
- **Pattern**: Released in `ensure` block after data extraction

### Provider Options (CUDA, TensorRT, etc.)

- **Creation**: Provider-specific API (e.g., `api.create_cuda_provider_options`)
- **Release**: Provider-specific release (e.g., `api.release_cuda_provider_options`)
- **Tracking**: `@released` flag in provider classes
- **Pattern**: Released via finalizer or explicit `release` method

### OrtIoBinding

- **Creation**: `api.create_io_binding`
- **Release**: `api.release_io_binding`
- **Tracking**: `@released` flag
- **Pattern**: Released via finalizer or explicit `release` method

## Common Pitfalls

### 1. Double-Free Errors

**Problem**: Resource released by both Crystal finalizer and C library

**Solution**: Use `@resource_released` flags and check before releasing:

```crystal
def release_session
  return if @session_released
  api.release_session.call(@session) if @session
  @session_released = true
end
```

### 2. Memory Leaks from Exceptions

**Problem**: Exception thrown before resource release, causing leak

**Solution**: Always use `begin/ensure`:

```crystal
memory_info = create_cpu_memory_info(api)
# ... use memory_info ...
ensure
  api.release_memory_info.call(memory_info) if memory_info && api
end
```

### 3. Dangling Pointers

**Problem**: Accessing ORT resource after it's been released

**Solution**: Track release state and raise on access:

```crystal
def env
  raise "Environment has been released" if @released
  @env
end
```

### 4. Resource Release Order

**Problem**: Releasing environment before sessions causes crashes

**Solution**: Always release in correct order:
1. Release all sessions first
2. Release environment last

```crystal
session1.release
session2.release
OnnxRuntime::InferenceSession.release_env  # Must be last
```

### 5. Thread Safety Issues

**Problem**: Concurrent access to shared environment without synchronization

**Solution**: Use mutex in `OrtEnvironment`:

```crystal
@@mutex.synchronize do
  @@instance ||= new
end
```

## Development Best Practices

### Adding New ORT API Bindings

When adding new ONNX Runtime API functions:

1. **Add function pointer to `OrtApi` struct** in `libonnxruntime.cr`
2. **Define type aliases** for any new ORT types
3. **Create wrapper methods** in appropriate classes (`InferenceSession`, `Tensor`, etc.)
4. **Implement proper resource tracking** with `@released` flags if applicable
5. **Use `ensure` blocks** to guarantee cleanup
6. **Test resource release** to verify no leaks or double-frees

### Error Handling

Always check API call status and release status object:

```crystal
private def check_status(status)
  return if status.null?

  error_code = api.get_error_code.call(status)
  error_message = String.new(api.get_error_message.call(status))
  api.release_status.call(status)  # Important: release status

  raise "ONNXRuntime Error: #{error_message} (#{error_code})"
end
```

### Testing Resource Management

When writing tests, always ensure proper cleanup:

```crystal
Spec.after_suite do
  OnnxRuntime::OrtEnvironment.instance.release
end
```

## Architecture Summary

```
src/onnxruntime/
├── libonnxruntime.cr       # C API bindings (OrtApi function pointers)
├── ort_environment.cr      # OrtEnv singleton manager
├── inference_session.cr    # OrtSession wrapper (inference entry point)
├── tensor.cr               # OrtValue wrapper for dense tensors
├── sparse_tensor.cr        # OrtValue wrapper for sparse tensors
├── tensor_info.cr          # Tensor metadata (shape, type)
├── model_metadata.cr       # Model metadata reader
├── provider.cr             # Execution provider options (CPU, CUDA, etc.)
├── run_options.cr          # OrtRunOptions wrapper
├── io_binding.cr           # OrtIoBinding wrapper (zero-copy I/O)
└── training_session.cr     # Training API (placeholder)
```

### Design Philosophy

- **Explicit over implicit**: Require manual resource release for predictability
- **Fail fast**: Raise errors on invalid states (e.g., accessing released resources)
- **Minimize C<->Crystal boundary crossings**: Batch operations when possible
- **Safety first**: Prefer ensure blocks and state tracking over relying on finalizers
