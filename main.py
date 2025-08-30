import tensorflow as tf

# Check if TensorFlow is working and print its version
print(f"TensorFlow Version: {tf.__version__}")

# A simple example to prove it's installed and functional
hello_world = tf.constant("Hello, TensorFlow!")
print(hello_world)