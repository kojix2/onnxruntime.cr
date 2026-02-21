require "./mnist_dataset"

# MNIST Single Image Prediction Example
# This example demonstrates how to use the ONNXRuntime.cr library to predict a single MNIST digit

# Initialize dataset and model
config = MNISTDataset.initialize_dataset
model = MNISTDataset.load_model(config.model_path)
predict_single = MNISTDataset.create_prediction_function(model)

# Choose a random test image
image_index = Random.rand(MNISTDataset::MNIST_TEST_COUNT)
puts "\nReading MNIST test image ##{image_index}"

# Read image and label
begin
  input_data = MNISTDataset::Dataset.read_image(config.images_file, image_index)
  true_label = MNISTDataset::Dataset.read_label(config.labels_file, image_index)
  puts "True label: #{true_label}"
rescue ex
  puts "Error reading MNIST data: #{ex.message}"
  puts "Using a simulated MNIST image instead."

  input_data = MNISTDataset::Dataset.create_simulated_digit_3
  true_label = 3
  puts "True label: #{true_label} (simulated)"
end

# Display input shape information
puts "\nInput data shape: [1, 1, #{MNISTDataset::MNIST_IMAGE_SIZE}, #{MNISTDataset::MNIST_IMAGE_SIZE}]"
puts "Total input elements: #{input_data.size}"

# Run inference
puts "\nRunning inference..."
result = predict_single.call(input_data)
prediction = result[:prediction]
scores = result[:scores]

# Display prediction results
puts "\nPrediction results (raw scores):"
scores.each_with_index do |score, i|
  puts "  - Digit #{i}: #{score}"
end

# Display the class with highest probability
puts "\nPredicted digit: #{prediction} with confidence: #{scores[prediction]}"
puts "True digit: #{true_label}"
puts "Prediction #{prediction == true_label ? "CORRECT" : "INCORRECT"}"

# Visualize the image
MNISTDataset::Dataset.visualize_image(input_data)

# Explicitly release resources
model.release
OnnxRuntime::InferenceSession.release_env
