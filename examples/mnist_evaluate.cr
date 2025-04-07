require "./mnist_dataset"
require "option_parser"

# MNIST Evaluation Example
# This example demonstrates how to evaluate the MNIST model on multiple test images
# and calculate various performance metrics

# Parse command line options
evaluate_all = false
sample_count = 100

OptionParser.parse do |parser|
  parser.banner = "Usage: crystal examples/mnist_evaluate.cr [options]"

  parser.on("-a", "--all", "Evaluate all test images") do
    evaluate_all = true
  end

  parser.on("-n COUNT", "--count=COUNT", "Number of samples to evaluate") do |count|
    sample_count = count.to_i
  end

  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

# Initialize dataset and model
config = MNISTDataset.initialize_dataset
model = MNISTDataset.load_model(config.model_path)
predict_single = MNISTDataset.create_prediction_function(model)

# Determine number of images to evaluate
count = evaluate_all ? MNISTDataset::MNIST_TEST_COUNT : sample_count
count = [count, MNISTDataset::MNIST_TEST_COUNT].min

puts "\nEvaluating #{count} MNIST test images..."

# Arrays to store predictions and true labels
predictions = [] of Int32
true_labels = [] of Int32

# Progress tracking
progress_step = [count // 10, 1].max.to_i32

# Process each image
count.times do |i|
  # Show progress
  if i % progress_step == 0
    print "\rProcessing image #{i}/#{count} (#{(i * 100 / count).round(1)}%)..."
    STDOUT.flush
  end

  begin
    # Read image and label
    image_data = MNISTDataset::Dataset.read_image(config.images_file, i)
    true_label = MNISTDataset::Dataset.read_label(config.labels_file, i)

    # Make prediction
    result = predict_single.call(image_data)
    prediction = result[:prediction]

    # Store results
    predictions << prediction
    true_labels << true_label
  rescue ex
    puts "\nError processing image #{i}: #{ex.message}"
  end
end

puts "\rProcessed #{count} images (100%)                "

# Calculate metrics
confusion = MNISTDataset::Metrics.confusion_matrix(predictions, true_labels)
accuracy_value = MNISTDataset::Metrics.accuracy(predictions, true_labels)
precision_values = MNISTDataset::Metrics.precision(confusion)
recall_values = MNISTDataset::Metrics.recall(confusion)
f1_values = MNISTDataset::Metrics.f1_score(precision_values, recall_values)

# Display metrics
MNISTDataset::Metrics.display_confusion_matrix(confusion)
MNISTDataset::Metrics.display_metrics(precision_values, recall_values, f1_values, accuracy_value)

# Explicitly release resources
model.release
OnnxRuntime::InferenceSession.release_env
