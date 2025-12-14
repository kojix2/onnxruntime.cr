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
        producer_name = get_metadata_string(api, metadata, session.allocator, session) do |md, alloc, ptr|
          api.model_metadata_get_producer_name.call(md, alloc, ptr)
        end

        # Get graph name
        graph_name = get_metadata_string(api, metadata, session.allocator, session) do |md, alloc, ptr|
          api.model_metadata_get_graph_name.call(md, alloc, ptr)
        end

        # Get domain
        domain = get_metadata_string(api, metadata, session.allocator, session) do |md, alloc, ptr|
          api.model_metadata_get_domain.call(md, alloc, ptr)
        end

        # Get description
        description = get_metadata_string(api, metadata, session.allocator, session) do |md, alloc, ptr|
          api.model_metadata_get_description.call(md, alloc, ptr)
        end

        # Get graph description
        graph_description = get_metadata_string(api, metadata, session.allocator, session) do |md, alloc, ptr|
          api.model_metadata_get_graph_description.call(md, alloc, ptr)
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
      value_ptr = Pointer(Pointer(UInt8)).malloc(1)
      status = yield metadata, allocator, value_ptr

      # If the status indicates an error, return empty string
      if !status.null?
        api.release_status.call(status)
        return ""
      end

      value = value_ptr.value.null? ? "" : String.new(value_ptr.value)
      value
    end

    # Helper method to get string from metadata
    private def self.get_string_from_metadata(api, metadata, key, session)
      allocator = session.allocator
      value_ptr = Pointer(Pointer(UInt8)).malloc(1)

      status = api.model_metadata_lookup_custom_metadata_map.call(
        metadata,
        key.to_unsafe,
        allocator,
        value_ptr
      )

      # If key doesn't exist, return empty string
      if !status.null?
        api.release_status.call(status)
        return ""
      end

      value = value_ptr.value.null? ? "" : String.new(value_ptr.value)
      value
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
      keys = Pointer(Pointer(Pointer(UInt8))).malloc(1)

      # Get custom metadata keys count
      status = api.model_metadata_get_custom_metadata_map_keys.call(
        metadata,
        allocator,
        keys,
        pointerof(keys_count)
      )
      session.check_status(status)

      # Create hash to store custom metadata
      custom_metadata = {} of String => String

      # Get custom metadata values
      if keys_count > 0
        keys_array = keys.value
        keys_count.times do |i|
          key = String.new(keys_array[i])
          value = get_string_from_metadata(api, metadata, key, session)
          custom_metadata[key] = value
        end
      end

      custom_metadata
    end
  end
end
