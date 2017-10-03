# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./DataDistribution.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

### Data Set Summary & Exploration

#### 1. Data Summary

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 3

#### 2. Exploration

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed over the different classes. It also shows that the distributions of the training and test data are very similar with is important for the validity of the test scores.

![alt text][image1]

### Design and Test a Model Architecture

I decided not work with the images as normal RGB images as colors are import for many of the signs and this is data that I think would benefit the model. The only form of preprocessing I did was to use Mean subtraction and Normalization.

I chose not to do any data augmentation as I wanted to see how good of a model I can make only based on the provided data.



#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5    	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 1x1    	| 1x1 stride, same padding, outputs 32x32x128 	|
| RELU					|												|
| Convolution 3x3    	| 1x1 stride, same padding, outputs 32x32x256 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x256 				|
| Convolution 3x3    	| 1x1 stride, same padding, outputs 16x16x512 	|
| RELU					|												|
| Convolution 1x1    	| 1x1 stride, same padding, outputs 16x16x256 	|
| RELU					|												|
| Convolution 3x3    	| 1x1 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128 			    	|
| Fully connected		| with an output of 1024      					|
| Fully connected		| with an output of 512      					|
| Fully connected		| with an output of 256      					|
| Softmax				| for the 43 classes       						|
|						|												|
|:---------------------:|:---------------------------------------------:| 
 


#### 3. Hyperparameters

To train the model, I used an AdamOptimizer with batch size of 128 and I trained it for 30 epochs with a learning rate of 0.0003.

#### 4. Training Aproach

My final model results were:
* validation set accuracy of 95% 
* test set accuracy of 95.2%

At first I began with an version of LeNet as it was a stable architecture that I had experiance with. I got to 93% test accuracy with it and then I decided to try out other architectures. 

I experimented with different ideas based on various papers. I made mock implementation of the architectures proposed in the papers as those architectures were much bigger than what I needed. 

In the end I made my own custom architecture that combined some of those ideas. After that I mostly tuned the number of weights per layer, the size of the filter and the learning rate.

### Test a Model on New Images

After I finish training and i was satisfied with the results, the model was tested on some images taken from the Web. The images were taken from https://github.com/NikolasEnt/Traffic-Sign-Classifier. There are 12 images in this new data set. They are of different signs and are made in different coditions (e.g. some are shady, some are weirdly lit). Although there are images in the original data set that are made in similar conditions, there are features in all of these imageas that make their clasification non-trivial.

After testing the model on them, it achieved 50% success rate. It was missclassifing all of the speed limit signs as only 1 class (speed limit 30). Apperantly the model is highly biased towards the speed limit 30 class, as after testing the top 5 probabilities of the other images, this class was present in almost all of the circular traffic signs. This can potentially be fixes with more data/ data augmentation. 