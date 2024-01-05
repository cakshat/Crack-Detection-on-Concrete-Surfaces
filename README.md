# Crack Detection on Concrete Surfaces

This repository contains a Jupyter notebook that demonstrates how to use machine learning for crack detection on concrete surfaces. The model is trained on a dataset of images of concrete surfaces, some of which contain cracks.

## Code Breakdown
1. 	It imports the `InceptionV3` model from `tensorflow.keras.applications`.
2.	It sets the path to the local pre-trained weights file for the InceptionV3 model.
3.	It creates an instance of the InceptionV3 model with the following specifications:
  a.	The input shape for the model is set to `(150, 150, 3)`, which means it expects images of size 150x150 with 3 color channels (RGB).
  b.	The `include_top` parameter is set to `False`, which means the final fully connected layer of the model, responsible for classification, is not included. This allows you to add your own classification layers suitable for your specific problem.
  c.	The `weights` parameter is set to `None`, which means no weights are loaded initially.
4.	It loads the pre-trained weights into the InceptionV3 model from the local weights file.
5.	It sets all layers in the pre-trained model to be non-trainable. This is done because the weights of these layers, which have been pre-trained on a large dataset (ImageNet), are already good at extracting features from images, and we don't want to change them during training.
6.	It retrieves the output of the specified layer from the pre-trained model. This layer is chosen as the cutoff for the pre-trained layers.
7.	It flattens the output of the layer to 1D. This is necessary because the following dense layer expects input in this format.
8.	It adds a fully connected (dense) layer with 1024 hidden units and ReLU activation. This layer can learn to represent abstract features from the output layer.
9.	It adds a dropout layer that randomly sets 20% of the input units to 0 during training. This helps prevent overfitting.
10.	It adds a final dense layer with a single unit and sigmoid activation. This layer will output the predicted probability that the input image contains a crack.
11.	It creates a new model that includes the input and output layers of the pre-trained model, as well as the new layers.
12.	It compiles the model with the RMSprop optimizer, binary cross-entropy loss (appropriate for binary classification), and accuracy as the metric to monitor during training.
13.	Finally, it prints a summary of the new model, showing all layers, their types, and the number of parameters they have.

## Model used: InceptionV3
## InceptionV3 Model

In this project, the InceptionV3 model is used, a pre-trained convolutional neural network model for image classification. InceptionV3 is trained on more than a million images from the ImageNet database and can classify images into 1000 object categories. 
The InceptionV3 model architecture is characterized by its depth and complexity. It uses multiple kernel sizes in the same layer, allowing it to capture both high-level and low-level features in the images. This makes it particularly effective for complex image recognition tasks.
In this case, transfer learning is used to adapt the InceptionV3 model to the specific task of crack detection on concrete surfaces. Top layers of the pre-trained model are removed and replaced with a new fully connected layer that matches the number of classes in our problem. Then the model is trained on dataset, allowing it to learn from the specific features of our images.
By using the InceptionV3 model, we can leverage the power of a complex, pre-trained model, while adapting it to our specific task. This approach allows us to achieve high accuracy while reducing the amount of training data and computational resources required.
Three different choices have been tried for the cut-off layers: avg_pool (the beginning), mixed5 and mixed7 and these are the commonly used cut off layers.
|Cut-off Layer | Accuracy | Loss |
|-|-|-|
|`average_pooling2d`|0.9819|0.0526 |
|`mixed5`|0.9984|0.0043 |
|`mixed7`|0.9975|0.0085 |

