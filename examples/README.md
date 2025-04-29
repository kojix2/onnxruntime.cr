# Examples - ONNXRuntime.cr

This directory contains various sample programs using ONNXRuntime.cr.  
To run these examples, you need Crystal, the ONNX Runtime library, and the required model files.

## Prerequisites

- Crystal language is installed
- ONNX Runtime library is installed  
  e.g. [ONNX Runtime Release](https://github.com/microsoft/onnxruntime/releases)
- Set the environment variable `ONNXRUNTIME_DIR`  
  Example:
  ```sh
  export ONNXRUNTIME_DIR="/home/kojix2/Cpp/onnxruntime-linux-x64-1.21.0"
  ```

## List of Examples

### 1. mnist_single.cr

Run inference on a single MNIST image.

```sh
cd examples
crystal mnist_single.cr
```

- Required model: `../spec/fixtures/mnist.onnx`
- Required data: Downloaded automatically

---

### 2. mnist_evaluate.cr

Evaluate the model on multiple MNIST images and print accuracy, confusion matrix, etc.

```sh
cd examples
crystal mnist_evaluate.cr -n 100   # Evaluate 100 images
crystal mnist_evaluate.cr --all    # Evaluate all test images
```

- Required model: `../spec/fixtures/mnist.onnx`
- Required data: Downloaded automatically

---

### 3. mnist_kemal.cr

A minimal web server for MNIST digit recognition. You can draw digits in your browser and get predictions.

```sh
cd examples
crystal mnist_kemal.cr
```

- The server will start at `http://localhost:3000`
- Required model: `../spec/fixtures/mnist.onnx`

---

### 4. bidaf.cr

Question answering example using the BiDAF (Bidirectional Attention Flow) model.

```sh
cd examples
crystal bidaf.cr -m bidaf-9.onnx
```

- Required model: `bidaf-9.onnx`  
  [ONNX Model Zoo (BiDAF)](https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/bidirectional_attention_flow)
  ```
  curl -L -o bidaf-9.onnx https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx
  ```

---

## Notes

- If the model file path does not match your environment, please edit the path in each example.
- Depending on your ONNX Runtime or Crystal version, additional setup may be required.
