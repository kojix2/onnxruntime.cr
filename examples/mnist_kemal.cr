require "../src/onnxruntime"
require "http/server"
require "json"

# MNIST Web Application - Minimal Version
MNIST_SIZE = 28
MODEL_PATH = "spec/fixtures/mnist.onnx"

# HTML content
MNIST_HTML = <<-HTML
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>MNIST</title>
  <style>
    body { font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
    canvas { border: 1px solid #333; }
    button { margin: 5px; padding: 5px 10px; }
    .result { margin-top: 10px; font-weight: bold; }
    .bar-container { margin-top: 20px; }
    .bar-row { display: flex; align-items: center; margin-bottom: 4px; }
    .bar-label { width: 20px; text-align: right; margin-right: 8px; }
    .bar-outer { flex-grow: 1; background-color: #eee; height: 20px; border-radius: 2px; overflow: hidden; }
    .bar-inner { height: 100%; background-color: #4CAF50; transition: width 0.3s; }
    .bar-value { margin-left: 8px; min-width: 40px; }
  </style>
</head>
<body>
  <h1>MNIST Digit Recognition</h1>
  <canvas id="canvas" width="280" height="280"></canvas>
  <div>
    <button id="clear">Clear</button>
    <button id="predict">Predict</button>
  </div>
  <div id="result" class="result">Draw a digit and click Predict</div>
  <div id="bar-chart" class="bar-container"></div>

  <script>
    // Setup canvas
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Drawing state
    let drawing = false, lastX, lastY;
    
    // Drawing functions
    canvas.onmousedown = e => {
      drawing = true;
      [lastX, lastY] = [e.offsetX, e.offsetY];
    };
    
    canvas.onmousemove = e => {
      if (!drawing) return;
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(e.offsetX, e.offsetY);
      ctx.stroke();
      [lastX, lastY] = [e.offsetX, e.offsetY];
    };
    
    canvas.onmouseup = canvas.onmouseout = () => drawing = false;
    
    // Clear button
    document.getElementById('clear').onclick = () => {
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      document.getElementById('result').textContent = 'Draw a digit and click Predict';
      document.getElementById('bar-chart').innerHTML = ''; // Clear the bar chart
    };
    
    // Predict button
    document.getElementById('predict').onclick = async () => {
      // Get image data
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const pixels = imageData.data;
      
      // Resize to 28x28
      const blockSize = canvas.width / 28;
      const resizedData = new Array(28 * 28).fill(0);
      
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          let sum = 0, count = 0;
          
          // Average the pixels in this block
          for (let dy = 0; dy < blockSize; dy++) {
            for (let dx = 0; dx < blockSize; dx++) {
              const sx = Math.floor(x * blockSize + dx);
              const sy = Math.floor(y * blockSize + dy);
              if (sx < canvas.width && sy < canvas.height) {
                sum += 255 - pixels[(sy * canvas.width + sx) * 4]; // Invert: white->0, black->255
                count++;
              }
            }
          }
          
          // Normalize and threshold
          resizedData[y * 28 + x] = count > 0 && (sum / count) > 50 ? 1 : 0;
        }
      }
      
      try {
        // Send to server
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data: resizedData })
        });
        
        const result = await response.json();
        
        if (result.success) {
          // Display prediction result
          document.getElementById('result').textContent = 
            `Predicted: ${result.prediction} (Confidence: ${result.confidence.toFixed(2)})`;
          
          // Create bar chart
          const barChart = document.getElementById('bar-chart');
          barChart.innerHTML = '';
          
          // Convert scores to probabilities using softmax
          const softmax = (scores) => {
            const maxScore = Math.max(...scores);
            const exps = scores.map(score => Math.exp(score - maxScore));
            const sumExps = exps.reduce((sum, exp) => sum + exp, 0);
            return exps.map(exp => exp / sumExps);
          };
          
          // Calculate probabilities
          const probabilities = softmax(result.scores);
          
          // Create bars for each digit
          probabilities.forEach((probability, digit) => {
            // Calculate percentage for bar width (0-100%)
            const percentage = probability * 100;
            
            // Create row for this digit
            const row = document.createElement('div');
            row.className = 'bar-row';
            
            // Add digit label
            const label = document.createElement('div');
            label.className = 'bar-label';
            label.textContent = digit;
            row.appendChild(label);
            
            // Add bar container
            const barOuter = document.createElement('div');
            barOuter.className = 'bar-outer';
            
            // Add actual bar
            const barInner = document.createElement('div');
            barInner.className = 'bar-inner';
            barInner.style.width = `${percentage}%`;
            barOuter.appendChild(barInner);
            row.appendChild(barOuter);
            
            // Add score value
            const value = document.createElement('div');
            value.className = 'bar-value';
            value.textContent = probability.toFixed(4);
            row.appendChild(value);
            
            // Highlight the predicted digit
            if (digit === result.prediction) {
              row.style.fontWeight = 'bold';
              barInner.style.backgroundColor = '#2196F3';
            }
            
            // No midpoint indicator needed for probability display
            barOuter.style.position = 'relative';
            
            // Add row to chart
            barChart.appendChild(row);
          });
        } else {
          document.getElementById('result').textContent = `Error: ${result.error}`;
        }
      } catch (error) {
        document.getElementById('result').textContent = `Error: ${error.message}`;
      }
    };
  </script>
</body>
</html>
HTML

# Load model
puts "Loading MNIST model from #{MODEL_PATH}"
model = OnnxRuntime::Model.new(MODEL_PATH)

# Create HTTP server
server = HTTP::Server.new do |context|
  case {context.request.method, context.request.path}
  when {"GET", "/"}
    context.response.content_type = "text/html"
    context.response.print MNIST_HTML
  when {"POST", "/predict"}
    begin
      # Parse request
      body = context.request.body.try &.gets_to_end
      next unless body

      json_data = JSON.parse(body)
      pixel_data = json_data["data"].as_a.map { |v| v.as_i.to_f32 }

      # Print pattern visualization
      puts "Input pattern:"
      (0...MNIST_SIZE).each do |y|
        puts (0...MNIST_SIZE).map { |x| pixel_data[y * MNIST_SIZE + x] > 0.5 ? "X" : "." }.join
      end

      # Make prediction
      result = model.predict(
        {"Input3" => pixel_data},
        nil,
        shape: {"Input3" => [1_i64, 1_i64, MNIST_SIZE.to_i64, MNIST_SIZE.to_i64]}
      )

      # Get output
      output = result["Plus214_Output_0"].as(Array(Float32))
      prediction = output.index(output.max) || 0

      # Return result
      context.response.content_type = "application/json"
      context.response.print({
        success:    true,
        prediction: prediction,
        confidence: output.max,
        scores:     output,
      }.to_json)
    rescue ex
      puts "Error: #{ex.message}"
      context.response.status = HTTP::Status::BAD_REQUEST
      context.response.print({success: false, error: ex.message}.to_json)
    end
  else
    context.response.status = HTTP::Status::NOT_FOUND
    context.response.print "Not Found"
  end
end

# Start server
address = server.bind_tcp 3000
puts "Server running at http://localhost:3000"
server.listen
