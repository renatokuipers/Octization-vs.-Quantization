import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


def float_to_quantized(value, num_bits):
    max_val = 2**num_bits - 1
    quantized_val = int(value * max_val)
    return quantized_val

def quantize_weights(weights, num_bits=8):
    max_val = 2**num_bits - 1
    quantized_weights = np.round(weights * max_val).astype(np.int32)
    return quantized_weights

class QuantizeLayer(tf.keras.layers.Layer):
    def __init__(self, num_bits=8):
        super(QuantizeLayer, self).__init__()
        self.num_bits = num_bits

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape=[int(input_shape[-1]), 10], initializer='uniform')

    def call(self, inputs):
        quantized_kernel = float_to_quantized_tensor(self.kernel, self.num_bits)
        quantized_kernel = tf.cast(quantized_kernel, tf.float32)
        return tf.matmul(inputs, quantized_kernel)

def float_to_quantized_tensor(value, num_bits):
    max_val = tf.cast(2**num_bits - 1, tf.float32)
    quantized_val = tf.cast(value * max_val, tf.int32)
    return quantized_val


def float_to_octal_tensor(value, num_bits):
    max_val = tf.cast(2**num_bits - 1, tf.float32)
    octal_val = tf.cast(value * max_val, tf.int32)
    return octal_val

class OctalQuantizeLayer(tf.keras.layers.Layer):
    def __init__(self, num_bits=8):
        super(OctalQuantizeLayer, self).__init__()
        self.num_bits = num_bits

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape=[int(input_shape[-1]), 10], initializer='uniform')

    def call(self, inputs):
        quantized_kernel = float_to_octal_tensor(self.kernel, self.num_bits)
        quantized_kernel = tf.cast(quantized_kernel, tf.float32)
        return tf.matmul(inputs, quantized_kernel)


# Standard Quantization Model
quant_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    QuantizeLayer(num_bits=8),
    Dense(10, activation='softmax')
])

quant_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n###############################")
print("Standard Quantization Model")
start_time = time.time()
quant_history = quant_model.fit(x_train, y_train, epochs=20, validation_split=0.2)
quant_inference_time = time.time() - start_time
print("###############################\n")

# Octization Model
oct_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    OctalQuantizeLayer(num_bits=8),
    Dense(10, activation='softmax')
])

oct_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n###############################")
print("Octization Model")
start_time = time.time()
oct_history = oct_model.fit(x_train, y_train, epochs=20, validation_split=0.2)
oct_inference_time = time.time() - start_time
print("###############################\n")


# Evaluate Standard Quantization Model
quant_test_loss, quant_test_acc = quant_model.evaluate(x_test, y_test, verbose=2)
quant_model_size = sum(
    np.prod(v.shape.as_list()) for v in quant_model.trainable_weights
)

# Evaluate Octization Model
oct_test_loss, oct_test_acc = oct_model.evaluate(x_test, y_test, verbose=2)
oct_model_size = sum(
    np.prod(v.shape.as_list()) for v in oct_model.trainable_weights
)

# Print Results
print("\n##############################")
print(f"Standard Quantization - Test Accuracy: {quant_test_acc}, Test Loss: {quant_test_loss}, Inference Time: {quant_inference_time}, Model Size: {quant_model_size}")
print(f"Octization - Test Accuracy: {oct_test_acc}, Test Loss: {oct_test_loss}, Inference Time: {oct_inference_time}, Model Size: {oct_model_size}")
print("##############################\n")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(quant_history.history['accuracy'])
plt.plot(quant_history.history['val_accuracy'])
plt.title('Standard Quantization Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(oct_history.history['accuracy'])
plt.plot(oct_history.history['val_accuracy'])
plt.title('Octization Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()