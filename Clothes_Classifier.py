# Description: This program classifies clothes from the Fashion MNIST data set using artificial neural networks.

# Import the libraries.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# Load the data set
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# View a training image
img_index = 0
img = train_images[img_index]
print("Image Label: ", train_labels[img_index])
plt.imshow(img)
print(plt.show())


# Print the shape
print(train_images.shape)
print(test_images.shape)

# Create the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Model
model.fit(train_images, train_labels, epochs=5, batch_size=32)

# Evaluate the Model
model.evaluate(test_images, test_labels)

# Make a prediction/classification
predictions = model.predict(test_images[0:5])

# Print the predicted labels
print(np.argmax(predictions, axis=1))

# Print the actual label values
print(test_labels[0:5])

# Print the first 5 images
for i in range(0, 5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(test_images[i], cmap='gray')
    plt.show()
