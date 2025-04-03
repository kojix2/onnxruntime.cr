require "../src/onnxruntime"
require "file_utils"

# MNIST Dataset handling module
module MNISTDataset
  extend self

  # Constants
  MNIST_IMAGE_SIZE        = 28
  MNIST_PIXEL_COUNT       = MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE
  MNIST_IMAGE_HEADER_SIZE =    16
  MNIST_LABEL_HEADER_SIZE =     8
  MNIST_TEST_COUNT        = 10000
  MNIST_CLASS_COUNT       =    10

  # Configuration
  class Config
    property data_dir : String
    property model_path : String
    property images_file : String
    property labels_file : String
    property images_gz : String
    property labels_gz : String
    property images_url : String
    property labels_url : String

    def initialize
      @data_dir = "examples/data"
      @model_path = "spec/fixtures/mnist.onnx"
      @images_file = "#{@data_dir}/t10k-images-idx3-ubyte"
      @labels_file = "#{@data_dir}/t10k-labels-idx1-ubyte"
      @images_gz = "#{@data_dir}/t10k-images-idx3-ubyte.gz"
      @labels_gz = "#{@data_dir}/t10k-labels-idx1-ubyte.gz"
      @images_url = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
      @labels_url = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
    end
  end

  # Helper methods for dataset management
  module Dataset
    extend self

    # Download and extract a file using curl and gunzip
    def download_and_extract(url : String, gz_path : String, output_path : String, description : String) : Bool
      puts "Downloading MNIST #{description} from #{url}..."

      # Download using curl
      download_status = system("curl -s -L -o #{gz_path} #{url}")

      if download_status
        puts "Extracting MNIST #{description}..."
        # Extract using gunzip
        extract_status = system("gunzip -f #{gz_path}")

        if extract_status
          puts "#{description.capitalize} downloaded and extracted to #{output_path}"
          return true
        else
          puts "Error extracting #{description}. Using a simulated MNIST #{description} instead."
          # Create an empty file to avoid repeated download attempts
          File.write(output_path, "")
          return false
        end
      else
        puts "Error downloading MNIST #{description}. Using a simulated MNIST #{description} instead."
        # Create an empty file to avoid repeated download attempts
        File.write(output_path, "")
        return false
      end
    end

    # Create a simulated digit "3" image
    def create_simulated_digit_3 : Array(Float32)
      input_data = Array(Float32).new(MNIST_PIXEL_COUNT, 0.0_f32)

      # Draw the digit "3"
      # Horizontal line (top)
      (10..18).each do |x|
        input_data[MNIST_IMAGE_SIZE * 8 + x] = 1.0_f32
      end
      # Horizontal line (middle)
      (10..18).each do |x|
        input_data[MNIST_IMAGE_SIZE * 14 + x] = 1.0_f32
      end
      # Horizontal line (bottom)
      (10..18).each do |x|
        input_data[MNIST_IMAGE_SIZE * 20 + x] = 1.0_f32
      end
      # Vertical line (right)
      (8..20).each do |y|
        input_data[MNIST_IMAGE_SIZE * y + 18] = 1.0_f32
      end

      input_data
    end

    # Read an MNIST image from the dataset
    def read_image(file_path : String, index : Int32) : Array(Float32)
      if File.size(file_path) < MNIST_IMAGE_HEADER_SIZE + (index + 1) * MNIST_PIXEL_COUNT
        # File is empty or too small, create a simulated digit "3"
        return create_simulated_digit_3
      end

      # Read image from IDX file
      File.open(file_path, "r") do |file|
        # Skip header
        file.skip(MNIST_IMAGE_HEADER_SIZE)

        # Skip to the desired image
        file.skip(index * MNIST_PIXEL_COUNT)

        # Read the image data
        bytes = Bytes.new(MNIST_PIXEL_COUNT)
        file.read(bytes)

        # Convert to Float32 and normalize to [0, 1]
        Array(Float32).new(MNIST_PIXEL_COUNT) do |i|
          bytes[i].to_f32 / 255.0_f32
        end
      end
    end

    # Read an MNIST label from the dataset
    def read_label(file_path : String, index : Int32) : Int32
      if File.size(file_path) < MNIST_LABEL_HEADER_SIZE + index + 1
        # File is empty or too small, return a default label
        return 3
      end

      # Read label from IDX file
      File.open(file_path, "r") do |file|
        # Skip header
        file.skip(MNIST_LABEL_HEADER_SIZE)

        # Skip to the desired label
        file.skip(index)

        # Read the label
        byte = Bytes.new(1)
        file.read(byte)

        byte[0].to_i32
      end
    end

    # Visualize an MNIST image in the console
    def visualize_image(image_data : Array(Float32), size : Int32 = MNIST_IMAGE_SIZE)
      puts "\nInput image visualization:"
      size.times do |y|
        line = ""
        size.times do |x|
          pixel_value = image_data[size * y + x]
          line += pixel_value > 0.5 ? "██" : "  "
        end
        puts line
      end
    end
  end

  # Metrics calculation and display
  module Metrics
    extend self

    # Calculate confusion matrix
    def confusion_matrix(predictions : Array(Int32), labels : Array(Int32), num_classes : Int32 = MNIST_CLASS_COUNT) : Array(Array(Int32))
      matrix = Array.new(num_classes) { Array.new(num_classes, 0) }

      predictions.zip(labels).each do |pred, label|
        matrix[label][pred] += 1
      end

      matrix
    end

    # Calculate accuracy
    def accuracy(predictions : Array(Int32), labels : Array(Int32)) : Float64
      correct = predictions.zip(labels).count { |pred, label| pred == label }
      correct.to_f / predictions.size
    end

    # Calculate precision for each class
    def precision(confusion : Array(Array(Int32))) : Array(Float64)
      precision_values = Array.new(confusion.size, 0.0)

      confusion.size.times do |i|
        # True positives for class i
        true_positives = confusion[i][i]
        # Sum of all predictions for class i (column sum)
        total_predicted = confusion.sum { |row| row[i] }

        precision_values[i] = total_predicted > 0 ? true_positives.to_f / total_predicted : 0.0
      end

      precision_values
    end

    # Calculate recall for each class
    def recall(confusion : Array(Array(Int32))) : Array(Float64)
      recall_values = Array.new(confusion.size, 0.0)

      confusion.size.times do |i|
        # True positives for class i
        true_positives = confusion[i][i]
        # Sum of all actual instances of class i (row sum)
        total_actual = confusion[i].sum

        recall_values[i] = total_actual > 0 ? true_positives.to_f / total_actual : 0.0
      end

      recall_values
    end

    # Calculate F1 score for each class
    def f1_score(precision_values : Array(Float64), recall_values : Array(Float64)) : Array(Float64)
      f1_values = Array.new(precision_values.size, 0.0)

      precision_values.size.times do |i|
        p = precision_values[i]
        r = recall_values[i]

        f1_values[i] = (p + r > 0) ? 2 * p * r / (p + r) : 0.0
      end

      f1_values
    end

    # Display confusion matrix
    def display_confusion_matrix(confusion : Array(Array(Int32)))
      puts "\nConfusion Matrix:"
      puts "   | " + (0...confusion.size).map { |i| " #{i} " }.join(" | ") + " | <- Predicted"
      puts "---+" + "-" * (confusion.size * 5 + 1)

      confusion.each_with_index do |row, i|
        print " #{i} | "
        row.each do |count|
          print count.to_s.rjust(3) + " | "
        end
        puts
      end
      puts "^"
      puts "Actual"
    end

    # Display metrics
    def display_metrics(precision_values : Array(Float64), recall_values : Array(Float64), f1_values : Array(Float64), accuracy_value : Float64)
      puts "\nMetrics by Class:"
      puts "Class | Precision | Recall | F1 Score"
      puts "------+-----------+--------+---------"

      precision_values.size.times do |i|
        puts "  #{i}   | #{precision_values[i].round(4).to_s.rjust(9)} | #{recall_values[i].round(4).to_s.rjust(6)} | #{f1_values[i].round(4).to_s.rjust(7)}"
      end

      # Calculate macro averages
      macro_precision = precision_values.sum / precision_values.size
      macro_recall = recall_values.sum / recall_values.size
      macro_f1 = f1_values.sum / f1_values.size

      puts "------+-----------+--------+---------"
      puts "Macro | #{macro_precision.round(4).to_s.rjust(9)} | #{macro_recall.round(4).to_s.rjust(6)} | #{macro_f1.round(4).to_s.rjust(7)}"
      puts "\nOverall Accuracy: #{(accuracy_value * 100).round(2)}%"
    end
  end

  # Initialize the dataset
  def initialize_dataset : Config
    config = Config.new

    # Create data directory if it doesn't exist
    FileUtils.mkdir_p(config.data_dir)

    # Remove existing empty files to force re-download
    [config.images_file, config.labels_file].each do |file|
      if File.exists?(file) && File.size(file) == 0
        File.delete(file)
      end
    end

    # Remove existing gz files
    [config.images_gz, config.labels_gz].each do |file|
      File.delete(file) if File.exists?(file)
    end

    # Download and extract dataset files if they don't exist
    unless File.exists?(config.images_file)
      Dataset.download_and_extract(config.images_url, config.images_gz, config.images_file, "test images")
    end

    unless File.exists?(config.labels_file)
      Dataset.download_and_extract(config.labels_url, config.labels_gz, config.labels_file, "test labels")
    end

    # Check if the model exists
    unless File.exists?(config.model_path)
      puts "Error: MNIST model not found at #{config.model_path}"
      puts "Please make sure the model file exists in the spec/fixtures directory."
      exit(1)
    end

    config
  end

  # Load the MNIST model
  def load_model(model_path : String) : OnnxRuntime::Model
    puts "Loading MNIST model from #{model_path}"
    model = OnnxRuntime::Model.new(model_path)

    # Display model input and output information
    puts "\nModel inputs:"
    model.inputs.each do |input|
      puts "  - #{input[:name]}: #{input[:type]} #{input[:shape]}"
    end

    puts "\nModel outputs:"
    model.outputs.each do |output|
      puts "  - #{output[:name]}: #{output[:type]} #{output[:shape]}"
    end

    model
  end

  # Create a prediction function
  def create_prediction_function(model : OnnxRuntime::Model)
    # MNIST image dimensions for the model
    batch_size = 1
    channels = 1
    height = MNIST_IMAGE_SIZE
    width = MNIST_IMAGE_SIZE

    # Function to predict a single image
    ->(image_data : Array(Float32)) {
      result = model.predict(
        {"Input3" => image_data},
        nil,
        shape: {"Input3" => [batch_size.to_i64, channels.to_i64, height.to_i64, width.to_i64]}
      )

      output = result["Plus214_Output_0"].as(Array(Float32))
      {
        prediction: output.index(output.max) || 0,
        scores:     output,
      }
    }
  end
end
