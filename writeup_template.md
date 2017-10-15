## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/data_summary.png "Data summary"
[image2]: ./examples/sign_preview.png "Signs preview"
[image3]: ./examples/preprocessing1.png "Preprocessing"
[image4]: ./examples/augmentation.png "Augmentation"
[image5]: ./examples/preprocessing.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/pr.png "Precision and Recall"
[image9]: ./examples/web.png ""
[image17]: ./examples/prediction.png ""
[image19]: ./examples/probabilities2.png "After model tunning"
[image20]: ./examples/conv1.png "Output of conv1 layer"
[image21]: ./examples/conv2.png "Output of conv2 layer"


**Rubric Points**
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
**Writeup / README**

1. Provide a Writeup / README that includes all the rubric points and how was addressed each one. 

Writeup is this document.

Here is a link to my [project code](https://github.com/greenfield932/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

**Data Set Summary & Exploration**

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed among the classes and a set of random images from training dataset
representing the dataset content.

![alt text][image1]
![alt text][image2]

**Design and Test a Model Architecture**

1. Image data preprocessing

To preprocess the data I used convertion images to grayscale using cvtColor function from opencv library and then applied normalization using exposure.equalize_adapthist function from skimage library.
Gray scale convertation was made to reduce amount of channels, and histogram normalization centers image data around it's mean value.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image3]

CNN have built-in invariance to small perturbations like rotations and transformations. I used data generation to extend original training set to make neural network more robust to deformations.

Here is an example of an original image and an augmented image:

![alt text][image4]

Result training dataset contains concatenation of original and augmented dataset (rotation, tranfsorm, rotation+transform per image from original dataset) and consists of 139196 images

2. Final model architecture.

My final model based on modification of LeNet architecture by adding dropout layers and increasing depth of convolutional and size of fully connected layers, consists of the following layers:

| Layer       		|     Description				| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 RGB image   				| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24			|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x24	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24			|
| Flatten		| input 5x5x24 output 600			|
| Fully connected	| input 600 output 150				|
| RELU			|						|
| Dropout		| keep_prob = 0.75				|
| Fully connected	| input 300 output 150				|
| RELU			|						|
| Dropout		| keep_prob = 0.75				|
| Fully connected	| input 150 output 43				|
 

3. Model training

To train model I used AdamOptimizer algorithm, batch size of 128 and 100 epochs with learning rate 0.001

4. Discussion of the model

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.987
* test set accuracy of 0.97

The initial architecture of the neural network was choosen LeNet. Using the original dataset with simple normalization ((pixel-128)/128) test set accuracy was 82%.
After changing normalization techiques, accuracy was about 92% for 50 epochs. I tried to change activation function to softmax which caused very poor 
training and validation accuracy, so I switched back to ReLU. There was a very poor precision and recall on multiple signs (about 0.6), 
so I decided to extend dataset with augmentation techniques. After trying 10, 50 and 100 epochs I finally got 96% accuracy on 100 epochs and extended training dataset.
But I got 92% accuracy on unseen signs from web, there was only 1 of 13 signs with wrong prediction: children crossing sign.
It was predicted as bicycle crossing and actually looks similar because of small resolution of input data. While precision and recall were high enough for both classes (>0.8).
I looked at convolutional layers output and seems there were no enough features to distinguish these signs. 
So I decided to extend convolutional layers depth to add more filters. Also I added dropout with keep probability parameter 0.75 to compensate increased depth and prevent overfitting.
After training of 100 epochs I finally got 97% of accuracy on test set and 100% accuracy on unseen signs from web.
Precision and recall also increased up to 0.9 for most of signs.

![alt text][image8]

Convolutional layer might work well for this problem because it provide robustness to deformations of the image data content, and was designed to extract features from images so it is a 
direct application for traffic signs classification problem.

This model architecture is suitable for this problem because based on LeNet architecture which was developed to recognize hand written digits. I believe this task 
is very close to traffic signs classification, except hand written digits is likely have less features than traffic signs, so modification of initial architecture may increase
the model accuracy.

Important design choices: 
* preprocess and extend data to get better results on training and evaluating network
* check metrics (train, validation, test accuracy, precision, recall, visualization of convolutional layers) to understand if network works well
* increase convolutional layers depth to add more filters
* add dropout to prevent overfitting in increased network
 
**Test a Model on New Images**

1. Overview of new data found on the web (maps.google.com, I used mostly Berlin roads)

![alt text][image9]

2. Models predictions

The results of the prediction:

![alt text][image17]

The model was able to correctly guess 12 of the 13 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97%.

3. Discussion on model certainty

As seen on top 5 probabilities for each new traffic sign image the model is quite certain. Mostly the model is confident about it's predictions, because the probability 
for all correct answers is very high relative to other probabilities.

![alt text][image19]

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.


**(Optional) Visualizing the Neural Network**
**1. Discuss the visual output of the trained network's feature maps.**

![alt text][image20]

As seen on output of convolutional layer 1 there are quite visible features used to recognize keep right sign. There are strong outlines of the sign, round edge, arrow.

![alt text][image21]

The arrow can be found even on second layer.

