# Convulution Neural Network
## Dataset
[MNIST dataset](https://s3.amazonaws.com/img-datasets/mnist.npz) is dataset of handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples.

The digits have been size-normalized and centered in a fixed-size image 28 * 28 pixels. For simplicity, each image has been flattened and converted to a 1-D numpy array of 784 features (28 * 28).
![MNIST dataset](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

```python
# Loading MNIST Dataset
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## Convolution

The convolution of `f` and `g` is written `f * g` and is defined as the integral of the product of the two functions after one is reflected about the y-axis and shifted.
![Convolution](https://upload.wikimedia.org/wikipedia/commons/6/6a/Convolution_of_box_signal_with_itself2.gif)

- Conv layers: consist of a set of filters, which you can think of as just 2d matrices of numbers. We can use an input image and a filter to produce an output image by convolving the filter with the input image. This consists of

### Convolution on Image
- Overlaying the filter on top of the image at some location.
- Performing element-wise multiplication between the values in the filter and their corresponding values in the image.
- Summing up all the element-wise products. This sum is the output value for the destination pixel in the output image.
- Repeating for all locations.

#### Padding
To have the output image as the same size as the input image, we add zeros around the image so we can overlay the filter in more places.

Here's an explanation of the components used in the provided code and how they contribute to the convolutional neural network (CNN) architecture:

# Model
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, verbose=1)
```

## Conv2D
`Conv2D` is a 2D convolutional layer in Keras. It applies a specified number of convolution filters to the input image. These filters are learned during the training process. The `(3, 3)` argument specifies the size of the convolutional kernel (filter), and `64` indicates the number of filters. The `relu` activation function is applied to introduce non-linearity to the model.

## BatchNormalization
`BatchNormalization` is a technique used to improve the training of deep neural networks. It normalizes the activations of each layer, i.e., it adjusts and scales the activations, ensuring that the mean activation close to zero and the standard deviation close to one. This helps in mitigating the internal covariate shift problem and accelerates the training process.

## MaxPooling2D
`MaxPooling2D` is a downsampling operation used in CNNs to reduce the spatial dimensions of the input volume. It extracts the maximum value from a subset of the input data. The `(2, 2)` argument specifies the size of the pooling window.

## Dropout
`Dropout` is a regularization technique used to prevent overfitting in neural networks. It randomly drops a fraction of input units during the training process, effectively "dropping out" neurons. This forces the network to learn more robust features and reduces the risk of overfitting.

## Flatten
`Flatten` is a layer that converts the output of the previous layer (which may be multi-dimensional) into a one-dimensional array. This is necessary when transitioning from convolutional layers to fully connected layers, as fully connected layers require one-dimensional input.

## Softmax
`Softmax` is an activation function used in the output layer of classification models. It converts the raw output scores (logits) of the model into probabilities. Each output neuron represents the probability of the corresponding class, and the sum of all probabilities is equal to one.

## ReduceLROnPlateau
`ReduceLROnPlateau` is a callback function in Keras that dynamically adjusts the learning rate during training. It monitors a specified metric (e.g., validation loss) and reduces the learning rate if the monitored quantity has stopped improving.

## EarlyStopping
`EarlyStopping` is another callback function in Keras. It stops the training process if the monitored quantity (e.g., validation loss) has stopped improving for a specified number of epochs. This helps prevent overfitting and saves training time.

## ModelCheckpoint
`ModelCheckpoint` is a callback function that saves the model's weights during training. It monitors a specified metric (e.g., validation loss) and saves the model's weights to a file whenever the monitored metric improves.

# Training
```python
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[reduce_lr, early_stop, checkpoint])
```

## Training the Model
The `fit` method is used to train the model on the training data. The `x_train` and `y_train` arguments represent the input images and their corresponding labels, respectively. The `epochs` parameter specifies the number of training iterations. The `validation_data` argument is used to evaluate the model's performance on the test data during training.

## Callbacks
The `callbacks` argument is used to specify a list of callback functions to be applied during training. In this case, the `reduce_lr`, `early_stop`, and `checkpoint` callbacks are used to dynamically adjust the learning rate, prevent overfitting, and save the best model weights, respectively.

## History
The `history` object stores the training history of the model, including the loss and accuracy values for each epoch. This information can be used to visualize the training progress and evaluate the model's performance.

# Evaluation
```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
```

## Evaluating the Model
The `evaluate` method is used to evaluate the model's performance on the test data. It returns the loss and accuracy values of the model on the test data. These metrics provide insights into how well the model generalizes to unseen data.

## Conclusion
In summary, the provided code demonstrates the implementation of a convolutional neural network (CNN) using Keras for the MNIST dataset. The model architecture consists of convolutional layers, batch normalization, max pooling, dropout, and dense layers with softmax activation. Callback functions are used to adjust the learning rate, prevent overfitting, and save the best model weights. The model is trained on the training data and evaluated on the test data to assess its performance. The training history and evaluation results provide valuable insights into the model's training progress and generalization capabilities.

# References
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)