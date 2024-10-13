import numpy as np
import matplotlib.pyplot as plt
import struct

# Function to load ubyte dataset
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # First 16 bytes are the header
        f.read(16)
        # Remaining bytes are the image pixels
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(-1, 28, 28)  # MNIST images are 28x28 pixels
    return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # First 8 bytes are the header
        f.read(8)
        # Remaining bytes are the labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load the dataset
images = load_mnist_images(r'B:\_GITHUB\domain_adaptive_thorax_disease_classification\pytorch-adda\data\MNIST\raw\train-images-idx3-ubyte')
labels = load_mnist_labels(r'B:\_GITHUB\domain_adaptive_thorax_disease_classification\pytorch-adda\data\MNIST\raw\t10k-images-idx3-ubyte')

# Visualize some images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
plt.show()
