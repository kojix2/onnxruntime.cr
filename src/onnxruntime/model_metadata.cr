module OnnxRuntime
  # ModelMetadata class provides access to ONNX model metadata.
  class ModelMetadata
    getter producer_name : String
    getter graph_name : String
    getter domain : String
    getter description : String
    getter graph_description : String
    getter version : Int64
    getter custom_metadata_map : Hash(String, String)

    # Creates a new ModelMetadata instance from a session.
    def self.from_session(session : InferenceSession)
      metadata = Pointer(LibOnnxRuntime::OrtModelMetadata).null
      api = session.api

      # Get model metadata
      status = api.session_get_model_metadata.call(session.session, pointerof(metadata))
      session.check_status(status)

      begin
        # Get producer name
        producer_name = get_metadata_string(api, metadata, session.allocator, session) do |metadata_ptr, allocator_ptr, value_ptr|
          api.model_metadata_get_producer_name.call(metadata_ptr, allocator_ptr, value_ptr)
        end

        # Get graph name
        graph_name = get_metadata_string(api, metadata, session.allocator, session) do |metadata_ptr, allocator_ptr, value_ptr|
          api.model_metadata_get_graph_name.call(metadata_ptr, allocator_ptr, value_ptr)
        end

        # Get domain
        domain = get_metadata_string(api, metadata, session.allocator, session) do |metadata_ptr, allocator_ptr, value_ptr|
          api.model_metadata_get_domain.call(metadata_ptr, allocator_ptr, value_ptr)
        end

        # Get description
        description = get_metadata_string(api, metadata, session.allocator, session) do |metadata_ptr, allocator_ptr, value_ptr|
          api.model_metadata_get_description.call(metadata_ptr, allocator_ptr, value_ptr)
        end

        # Get graph description
        graph_description = get_metadata_string(api, metadata, session.allocator, session) do |metadata_ptr, allocator_ptr, value_ptr|
          api.model_metadata_get_graph_description.call(metadata_ptr, allocator_ptr, value_ptr)
        end

        # Get version
        version = 0_i64
        status = api.model_metadata_get_version.call(metadata, pointerof(version))
        session.check_status(status)

        # Get custom metadata map
        custom_metadata_map = get_custom_metadata_map(api, metadata, session)

        new(producer_name, graph_name, domain, description, graph_description, version, custom_metadata_map)
      ensure
        # Release model metadata
        api.release_model_metadata.call(metadata) if metadata
      end
    end

    private def initialize(@producer_name, @graph_name, @domain, @description, @graph_description, @version, @custom_metadata_map)
    end

    # Helper method to get metadata string using API function
    private def self.get_metadata_string(api, metadata, allocator, session, &)
      value_ptr = Pointer(UInt8).null
      status = yield metadata, allocator, pointerof(value_ptr)

      # Return empty only when the metadata field is genuinely absent.
      if !status.null?
        error_code = api.get_error_code.call(status)
        error_message = String.new(api.get_error_message.call(status))
        api.release_status.call(status)

        return "" if error_code.not_found?

        raise "ONNXRuntime Error: #{error_message} (#{error_code})"
      end

      begin
        value_ptr.null? ? "" : String.new(value_ptr)
      ensure
        unless value_ptr.null?
          free_status = api.allocator_free.call(allocator, value_ptr.as(Void*))
          session.check_status(free_status)
        end
      end
    end

    # Helper method to get string from metadata
    private def self.get_string_from_metadata(api, metadata, key, session)
      allocator = session.allocator
      value_ptr = Pointer(UInt8).null

      status = api.model_metadata_lookup_custom_metadata_map.call(
        metadata,
        key.to_unsafe,
        allocator,
        pointerof(value_ptr)
      )

      # Return empty when a key is not present. Raise for other failures.
      if !status.null?
        error_code = api.get_error_code.call(status)
        error_message = String.new(api.get_error_message.call(status))
        api.release_status.call(status)

        return "" if error_code.not_found?

        raise "ONNXRuntime Error: #{error_message} (#{error_code})"
      end

      begin
        value_ptr.null? ? "" : String.new(value_ptr)
      ensure
        unless value_ptr.null?
          free_status = api.allocator_free.call(allocator, value_ptr.as(Void*))
          session.check_status(free_status)
        end
      end
    end

    # Helper method to get int64 from metadata
    private def self.get_int64_from_metadata(api, metadata, key, session)
      value_str = get_string_from_metadata(api, metadata, key, session)
      value_str.empty? ? 0_i64 : value_str.to_i64
    rescue
      0_i64
    end

    # Helper method to get custom metadata map
    private def self.get_custom_metadata_map(api, metadata, session)
      allocator = session.allocator
      keys_count = 0_u64
      keys = Pointer(Pointer(UInt8)).null

      # Get custom metadata keys count
      status = api.model_metadata_get_custom_metadata_map_keys.call(
        metadata,
        allocator,
        pointerof(keys),
        pointerof(keys_count)
      )
      session.check_status(status)

      # Create hash to store custom metadata
      custom_metadata = {} of String => String

      begin
        # Get custom metadata values
        if keys_count > 0 && !keys.null?
          keys_array = keys
          keys_count.times do |i|
            key_ptr = keys_array[i]

            begin
              key = key_ptr.null? ? "" : String.new(key_ptr)
              value = get_string_from_metadata(api, metadata, key, session)
              custom_metadata[key] = value
            ensure
              unless key_ptr.null?
                free_status = api.allocator_free.call(allocator, key_ptr.as(Void*))
                session.check_status(free_status)
              end
            end
          end
        end
      ensure
        unless keys.null?
          free_keys_status = api.allocator_free.call(allocator, keys.as(Void*))
          session.check_status(free_keys_status)
        end
      end

      custom_metadata
    end
  end
end
