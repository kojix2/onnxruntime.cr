# onnxruntime.cr

[![build](https://github.com/kojix2/onnxruntime.cr/actions/workflows/test.yml/badge.svg)](https://github.com/kojix2/onnxruntime.cr/actions/workflows/test.yml)

[ONNX Runtime](https://github.com/Microsoft/onnxruntime) bindings for Crystal

## Installation

1. Install ONNX Runtime

   Download and install the ONNX Runtime from the [official releases](https://github.com/microsoft/onnxruntime/releases).

   For Linux:
   ```bash
   # Example for Linux
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz
   tar -xzf onnxruntime-linux-x64-1.21.0.tgz
   export ONNXRUNTIMEDIR=/path/to/onnxruntime-linux-x64-1.21.0
   ```

   For macOS:
   ```bash
   # Example for macOS
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-osx-x86_64-1.21.0.tgz
   tar -xzf onnxruntime-osx-x86_64-1.21.0.tgz
   export ONNXRUNTIMEDIR=/path/to/onnxruntime-osx-x86_64-1.21.0
   ```

   For Windows:
   ```powershell
   # Example for Windows
   # Download from https://github.com/microsoft/onnxruntime/releases
   # Extract and set environment variable
   $env:ONNXRUNTIMEDIR = "C:\path\to\onnxruntime-win-x64-1.21.0"
   ```

2. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     onnxruntime:
       github: kojix2/onnxruntime.cr
   ```

3. Run `shards install`

## Usage

```crystal
require "onnxruntime"

# Load a model
model = OnnxRuntime::Model.new("path/to/model.onnx")

# Print model inputs and outputs
puts "Inputs:"
model.inputs.each do |input|
  puts "  #{input[:name]}: #{input[:type]} #{input[:shape]}"
end

puts "Outputs:"
model.outputs.each do |output|
  puts "  #{output[:name]}: #{output[:type]} #{output[:shape]}"
end

# Prepare input data
input_data = {
  "input_name" => [1.0_f32, 2.0_f32, 3.0_f32]
}

# Run inference
result = model.predict(input_data)

# Process results
result.each do |name, data|
  puts "#{name}: #{data}"
end
```

### Example with MNIST

```crystal
require "onnxruntime"

# Load the MNIST model
model = OnnxRuntime::Model.new("mnist.onnx")

# Create a dummy input (28x28 image with all zeros)
input_data = Array(Float32).new(28 * 28, 0.0_f32)

# Run inference
result = model.predict({"Input3" => input_data})

# Get the output probabilities
probabilities = result["Plus214_Output_0"].as(Array(Float32))

# Find the digit with highest probability
predicted_digit = probabilities.each_with_index.max_by { |prob, _| prob }[1]
puts "Predicted digit: #{predicted_digit}"
```

## Development

To generate the C API bindings:

```bash
cd scripts
docker build -t c2ffi .
wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.21.0/include/onnxruntime/core/session/onnxruntime_c_api.h
docker run --rm -v $(pwd):/data c2ffi onnxruntime_c_api.h > capi.json
```

## Contributing

1. Fork it (<https://github.com/kojix2/onnxruntime.cr/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request
