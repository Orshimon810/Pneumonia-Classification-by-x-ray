# Pneumonia-Classification-by-x-ray

In this project, we use Convolutional Neural Networks (CNNs) to classify chest X-ray images into different categories based on their pathology. We employ four models: one for binary classification of normal and pneumonia images, another for categorization into normal, viral pneumonia, and bacterial pneumonia, a third model that uses the categorical output for KNN classification, and a fourth model for anomaly detection, identifying pneumonia as an anomaly.

## Dataset (From Kaggle):
The dataset contains 5,856 validated Chest X-Ray images split into training and testing sets.
Images are labeled as (disease: NORMAL/BACTERIA/VIRUS).
In the training set, there are 3884 images labeled as pneumonia and 1349 images labeled as normal.
In the test set, there are 390 pneumonia images and 234 normal images.
In the validation set, there are 16 images, evenly distributed with 8 images labeled as pneumonia and 8 images labeled as normal.
For the first model, these 16 images were used for validation. However, for subsequent models, the validation set was drawn from the training set.

## First Model architecture (binary):
 The model architecture consists of several convolutional blocks followed by dense layers for binary classification of chest X-ray images into normal and pneumonia categories.
 
•	Block One: This block consists of a convolutional layer with 32 filters, each of size 3x3, using ReLU activation.
Batch normalization is applied to normalize the activations. Max pooling with a 2x2 pool size is used to downsample the spatial dimensions.

•	Block Two: The second block includes a convolutional layer with 64 filters, followed by batch normalization and max pooling.
Dropout with a rate of 0.2 is applied after the max pooling layer to reduce overfitting.

•	Block Three: This block contains a convolutional layer with 128 filters, batch normalization, and max pooling. Dropout with a rate of 0.3 is applied to reduce overfitting by randomly setting a fraction of input units to zero during training.

•	Block Four: The fourth block includes a convolutional layer with 256 filters, batch normalization, and max pooling. 
Dropout with a rate of 0.4 is applied after the max pooling layer to reduce overfitting.

•	Flatten and Dense Layers: After the convolutional blocks, the feature maps are flattened to a 1D vector and fed into a dense layer with 128 units and ReLU activation. Batch normalization and dropout with a rate of 0.5 are applied to the dense layer.

•	Final Layer (Output): The final layer is a dense layer with a single unit and sigmoid activation, which outputs the probability of the input image belonging to the pneumonia class.

Overall, this architecture aims to learn hierarchical representations of the input images, starting from low-level features (edges, textures) to high-level features (patterns, shapes) for effective classification. Batch normalization helps stabilize and accelerate the training process, while dropout reduces overfitting by introducing noise during training.

Hyperparameters:
•	Optimizer: Adam with learning rate 3e-5
•	Loss Function: Binary Crossentropy
•	Batch Size: 32
•	Epochs: 10

![image](https://github.com/Orshimon810/Pneumonia-Classification-by-x-ray/assets/127754114/2fac14bb-7d0f-430c-a3db-687eb37015d0)
![image](https://github.com/Orshimon810/Pneumonia-Classification-by-x-ray/assets/127754114/536f7562-3f58-4b4a-9673-770018b86302)


## Results on test set:
<img src="https://github.com/Orshimon810/Pneumonia-Classification-by-x-ray/assets/127754114/f62fd32e-07cb-426e-a007-b0a289ee5c1a" width="400" height = "400">


## Second model architecture (categorial classification):
The model consists of four convolutional blocks followed by dense layers for multi-class classification of chest X-ray images into normal, bacterial pneumonia, and viral pneumonia categories.

•	Block One: Convolutional layer with 32 filters, each of size 3x3, using ReLU activation.
Batch normalization is applied to normalize the activations.
Max pooling with a 2x2 pool size is used to downsample the spatial dimensions.

•	Block Two: Convolutional layer with 64 filters, followed by batch normalization and max pooling.

•	Block Three: Convolutional layer with 128 filters, batch normalization, and max pooling. Dropout with a rate of 0.2 is applied to reduce overfitting.

•	Block Four: Convolutional layer with 256 filters, batch normalization, and max pooling. Dropout with a rate of 0.3 is applied to reduce overfitting.

•	Flatten and Dense Layers: The feature maps are flattened to a 1D vector and fed into a dense layer with 128 units and ReLU activation. Dropout with a rate of 0.5 is applied to the dense layer.

•	Final Layer (Output): The final layer is a dense layer with three units and softmax activation, which outputs the probabilities of the input image belonging to each class.

Hyperparameters:

•	Optimizer: Adam optimizer with a learning rate of 1e-5 is used to minimize the categorical crossentropy loss function.

•	Loss Function: Categorical crossentropy is used as the loss function, suitable for multi-class classification problems.

•	Metrics: The model is evaluated based on accuracy, which measures the percentage of correctly classified images in the validation set.

•	Batch Size: The batch size is set to 32, indicating the number of images processed in each training step.

A larger batch size can lead to faster training but may require more memory.

•	Dropout Rates: Dropout is applied to the last two convolutional blocks and the dense layer with rates of 0.2, 0.3, and 0.5, respectively.
Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to zero during training.

•	Input Shape: The input shape of the images is (224, 224, 1), indicating a height and width of 224 pixels and a single channel for grayscale images.

•	Number of Classes: The model has three output units corresponding to the three classes (normal, bacterial pneumonia, viral pneumonia), and softmax activation is used to compute the probabilities for each class.

•	The model was trained for 10 epochs.

•	Callback: EarlyStopping.

![image](https://github.com/Orshimon810/Pneumonia-Classification-by-x-ray/assets/127754114/9706b4f3-4429-40ce-9c31-0a1ed7e4d2ca)
![image](https://github.com/Orshimon810/Pneumonia-Classification-by-x-ray/assets/127754114/db25fef5-3ffa-4ece-8c4e-7b4d3bce0ee5)


## Results on test set:
<img src="https://github.com/Orshimon810/Pneumonia-Classification-by-x-ray/assets/127754114/bb7bb4a5-7b39-4c6b-80d5-a1f93b4e7148" width="400" height="400">

## Third Model architecture (autoencoder):
The data processing prepares chest X-ray images to detect pneumonia as anomalies.
It trains the model using only normal images, treating pneumonia images as anomalies.
This approach helps in early detection of pneumonia cases, ensuring robustness across different types of pneumonia.
The autoencoder model comprises an encoder section with convolutional layers followed by a decoder section with transpose convolutional layers. It aims to compress input chest X-ray images into a latent space representation and reconstruct them, focusing on detecting anomalies, particularly pneumonia cases, as deviations from the reconstructed normal images.

Encoder Layers:
	The model includes convolutional layers with increasing filters (32, 64, 128, 256) for feature extraction.
	Batch normalization and max pooling are used for normalization and dimensionality reduction, respectively.
 
 Latent Space Representation:
The encoder compresses input images into a latent space representation, capturing essential features.

 Decoder Layers:
Transpose convolutional layers are used in the decoder to reconstruct the original input shape from the compressed representation.

Batch Normalization:
Batch normalization is applied after each layer to stabilize and accelerate training.

Final Output:
The final layer uses a sigmoid activation function to output reconstructed images with values between 0 and 1.

Hyperparameters: 
The autoencoder model is compiled using the Adam optimizer with a learning rate of 1e-3 and the mean squared error loss function.
During training, the model is fit to the training images with the following hyperparameters:

Epochs: 30 epochs, indicating the number of times the entire training dataset is passed forward and backward through the neural network.

Batch Size: 32, specifying the number of training examples utilized in one iteration.

Validation Data: Validation images are used to evaluate the model's performance on unseen data after each epoch.

Shuffle: The training data is shuffled at the beginning of each epoch to introduce randomness and prevent the model from memorizing the order of the training examples.

## Examples of constructed images:
![image](https://github.com/Orshimon810/Pneumonia-Classification-by-x-ray/assets/127754114/3eeaa220-c9a6-488a-9778-a342c0a5e767)

