**Build a Traffic Sign Recognition Project**

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
[image9]: ./unseen/9.png ""
[image10]: ./unseen/10.png ""
[image11]: ./unseen/11.png ""
[image12]: ./unseen/12.png ""
[image13]: ./unseen/13.png ""
[image14]: ./unseen/14.png ""
[image15]: ./unseen/15.png ""
[image16]: ./unseen/16.png ""
[image17]: ./examples/prediction.png ""
[image18]: ./examples/probabilities.png ""


**Rubric Points**
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
**Writeup / README**

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

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

The difference between the original data set and the augmented data set is: augmented dataset is 3 times bigger (rotation, tranfsorm, rotation+transform per image).
Result dataset contains concatenation of original and augmented.

2. Final model architecture.

My final model based on LeNet architecture and consisted of the following layers:

| Layer       		|     Description				| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 RGB image   				| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16			|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16			|
| Flatten		| input 5x5x16 output 400			|
| Fully connected	| input 400 output 120				|
| RELU			|						|
| Fully connected	| input 120 output 84				|
| RELU			|						|
| Fully connected	| input 84 output 43				|
 

3. Model training

To train model I used AdamOptimizer algorithm, batch size of 128 and 50 epochs with learning rate 0.001

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.972
* test set accuracy of 0.96

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

**Test a Model on New Images**

1. Here are five German traffic signs that I found on the web:

![alt text][image9] ![alt text][image10] ![alt text][image11] 
![alt text][image12] ![alt text][image13]

The first image might be difficult to classify because ...

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image18]

The model was able to correctly guess 12 of the 13 traffic signs, which gives an accuracy of 92%. This compares favorably to the accuracy on the test set of ...

3. Discussion in model certainty

As seen on top 5 probabilities for each new traffic sign image the model is quite certain. Mostly the model is confident in it's predictions because when answer is correct the probability 
for the answer is very high. When answer is wrong the are several high probabilities for different signs.

![alt text][image19]

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


