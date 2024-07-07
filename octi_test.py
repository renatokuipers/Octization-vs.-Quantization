import numpy as np
import tensorflow as tf

def float_to_octal(value, num_bits):
    """
    Convert a floating-point number to an octal-based discrete value.
    :param value: The floating-point number to convert.
    :param num_bits: Number of bits to represent the octal value.
    :return: Octal representation as a string.
    """
    max_val = 2**num_bits - 1
    octal_val = int(value * max_val)
    octal_str = oct(octal_val)
    return octal_str

# Example usage
float_value = 0.75
octal_representation = float_to_octal(float_value, 8)
print("\n############################")
print(f"Octal representation of {float_value} with 8 bits:\n{octal_representation}")
print("############################\n")

def quantize_to_octal(weights, num_bits=8):
    """
    Quantize weights to octal values.
    :param weights: Numpy array of weights.
    :param num_bits: Number of bits for quantization.
    :return: Quantized weights in octal representation.
    """
    quantized_weights = np.vectorize(lambda x: float_to_octal(x, num_bits))(weights)
    return quantized_weights

# Example: Quantizing weights of a neural network layer
layer = tf.keras.layers.Dense(10)
layer.build((None, 32))  # Build the layer to initialize weights
weights, biases = layer.get_weights()

# Quantize weights
quantized_weights = quantize_to_octal(weights)
quantized_biases = quantize_to_octal(biases)

print("\n############################")
print("Original weights:\n", weights)
print("Quantized weights:\n", quantized_weights)
print("############################\n")

def float_to_octal_tensor(value, num_bits):
    """
    Convert a floating-point tensor to an octal-based discrete tensor value.
    :param value: The floating-point tensor to convert.
    :param num_bits: Number of bits to represent the octal value.
    :return: Octal representation as a tensor.
    """
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

# Create a simple model with OctalQuantizeLayer
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32,)),
    OctalQuantizeLayer(num_bits=8),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Generate some random data
x_train = np.random.rand(100, 32).astype(np.float32)
y_train = np.random.rand(100, 1).astype(np.float32)

# Train the model
model.fit(x_train, y_train, epochs=50)

# Perform inference
x_test = np.random.rand(10, 32).astype(np.float32)
predictions = model.predict(x_test)
print("\n############################")
print("Predictions:\n", predictions)
print("############################\n")