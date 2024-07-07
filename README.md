```markdown
# Quantization and Octization Methodologies

This repository contains implementations and comparisons of two methodologies for quantizing neural network weights: Standard Quantization and a novel approach termed "Octization". The aim is to provide insights into their effectiveness in terms of accuracy, model size, inference time, and loss metrics.

## Methodologies

### Standard Quantization

Standard quantization converts floating-point weights into lower bit representations. Typically, the conversion is done using an 8-bit representation. The conversion function scales the weights to a discrete range, reducing the precision but potentially saving memory and speeding up computations.

**Code Snippet for Quantization:**

```python
def float_to_quantized(value, num_bits):
    max_val = 2**num_bits - 1
    quantized_val = int(value * max_val)
    return quantized_val

def quantize_weights(weights, num_bits=8):
    max_val = 2**num_bits - 1
    quantized_weights = np.round(weights * max_val).astype(np.int32)
    return quantized_weights
```

### Octization

Octization is a novel approach that represents weights in an octal system. This approach aims to offer a different granularity in quantization, which can sometimes provide better performance due to the unique representation scheme.

**Code Snippet for Octization:**

```python
def float_to_octal(value, num_bits):
    max_val = 2**num_bits - 1
    octal_val = int(value * max_val)
    octal_str = oct(octal_val)
    return octal_str

def quantize_to_octal(weights, num_bits=8):
    quantized_weights = np.vectorize(lambda x: float_to_octal(x, num_bits))(weights)
    return quantized_weights
```

## Results

### Quantization Results

**Quantized representation of 0.75 with 8 bits:**
```
191
```

**Original Weights:**
```
[[0.03279644, 0.37225065, 0.03883055, 0.32002673, 0.37683514, ...]]
```

**Quantized Weights:**
```
[[ 8,  95,  10,  82,  97, ...]]
```

**Training Metrics:**
```
Epoch 1/50: Loss: 382.4389
Epoch 50/50: Loss: 156.6920
```

**Predictions:**
```
[[ -1.8169721 ], [ 14.134164  ], [-10.419181  ], [ -9.413474  ], [ -7.104429  ], ...]
```

### Octization Results

**Octal representation of 0.75 with 8 bits:**
```
0o277
```

**Original Weights:**
```
[[0.03279644, 0.37225065, 0.03883055, 0.32002673, 0.37683514, ...]]
```

**Quantized Weights:**
```
[['0o10', '0o136', '0o11', '0o121', '0o140', ...]]
```

**Training Metrics:**
```
Epoch 1/50: Loss: 98.0795
Epoch 50/50: Loss: 42.9256
```

**Predictions:**
```
[[ 26.467916 ], [  1.4521854], [ 14.643928 ], [ -3.8951225], [-12.918487 ], ...]
```

## MNIST Dataset Results

### Standard Quantization

**Training:**
```
Epoch 1/20: Accuracy: 81.56%, Loss: 1.9228
Epoch 20/20: Accuracy: 99.36%, Loss: 0.0242
```

**Evaluation:**
```
Test Accuracy: 96.73%, Test Loss: 0.2253, Inference Time: 27.97s, Model Size: 101870
```

### Octization

**Training:**
```
Epoch 1/20: Accuracy: 80.83%, Loss: 2.0764
Epoch 20/20: Accuracy: 99.19%, Loss: 0.0288
```

**Evaluation:**
```
Test Accuracy: 96.93%, Test Loss: 0.1858, Inference Time: 27.99s, Model Size: 101870
```

## CIFAR-10 Dataset Results

### 5 Epochs

#### Standard Quantization

**Training:**
```
Epoch [1/5]: Loss: 2.2742
Epoch [5/5]: Loss: 2.1364
```

**Evaluation:**
```
Accuracy: 29.14%, Inference Time: 53.67s, Model Size: 282250
```

#### Octization

**Training:**
```
Epoch [1/5]: Loss: 2.1202
Epoch [5/5]: Loss: 1.7040
```

**Evaluation:**
```
Accuracy: 40.40%, Inference Time: 54.19s, Model Size: 282250
```

### 20 Epochs

#### Standard Quantization

**Training:**
```
Epoch [1/20]: Loss: 2.2671
Epoch [20/20]: Loss: 1.9121
```

**Evaluation:**
```
Accuracy: 33.32%, Inference Time: 214.58s, Model Size: 282250
```

#### Octization

**Training:**
```
Epoch [1/20]: Loss: 2.1327
Epoch [20/20]: Loss: 1.3204
```

**Evaluation:**
```
Accuracy: 51.89%, Inference Time: 217.92s, Model Size: 282250
```

## Conclusion

Both methodologies show promising results with Octization generally outperforming standard quantization in terms of accuracy across the board. However, the inference time and model size remain comparable between the two methods. The choice between these methods may depend on the specific application requirements, including the trade-offs between accuracy, model size, and computational efficiency.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow and PyTorch for providing the frameworks used in this project.

```

Feel free to clone the repository and run the provided scripts to further explore and analyze the quantization and octization methodologies.

```
