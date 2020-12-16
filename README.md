[//]: # (Image References)

[image1]: ./images/sample_dog_output_2.PNG "Sample Output"


## Classifying Dog Breeds using CNNs in PyTorch

### Project Overview

This project's mission is to build a machine learning pipeline that can be integrated in a web application to process real-world, user-supplied images. The created algorithm will complete two different tasks; Identify if a dog is in an image and output an estimate of the dog's breed, or identify if the image contains a human instead and output the closest resembling dog breed. In order to solve this multiclass classification problem, a convolutional neural network (CNN) can be implemented. This solution consists of three stages; dog detection, human detection, and implementing a CNN model for inference. The first two steps can use pretrained classifiers like OpenCV's Haar feature-based cascade classifier for detecting human faces, and the VGG16 model that is pretrained on the ImageNet dataset can detect dogs. After either a dog or human is detected, the last steps are to train a CNN on a dataset of images from the 133 dog breeds to classify and to pass the image to the trained CNN to process and output a prediction of the breed.

### CNN Model Creation

The CNN model I created in this project consists of 5 convolutional layers, each having a kernel size of 3x3 and stride of 1x1 then followed by a ReLu activation and MaxPooling layer. The input for the CNN is a 224x224 image. The pooling layers have a kernel size of 2x2 and therefore down-sample the image by 2 times after each one. After the 5 pooling layers the outputs are 7x7 feature maps. After the convolutional layers are the fully connected layers that output the scores of the classes predicted by the model. After training this model, it tested with 34% accuracy, which is acceptable for a CNN with a simple architecture and only run for 100 epochs.

### CNN Transfer Learning 

Designing a CNN architechture from scratch can be quite complicated. However, transfer learning allows for the use of some of the best pretrained CNN models to tune for the problem at hand. I have chosen to implement the ResNet-101 model, that is pretrained on the ImageNet dataset, as my transfer learning model. After changing the architecture to output 133 classes for the problem of classifying between the 133 dog breeds in this dataset, the model was trained and tested with an accuracy of 85%. This is a much improved performance over the simple architecture I previously created, and it was only run for 20 epochs.

![Sample Output][image1]

### Project Structure

1. Human face detector using the Haar feature-based cascade classifier
2. Dog detector using the VGG16 model pretrained on ImageNet dataset
3. Create CNN model architecture from scratch for dog breed classification.
4. Transfered pretrained ResNet-101 model architecture for dog breed classification.
5. Create algorithm to apply parts 1,2 & 4 to a given image and output breed prediction.
6. Algorithm testing 

Further detailed information on each step of the project can be found in the following notebook file.
	
	```
		jupyter notebook dog_app.ipynb
	```
### Packages Required to Run Project

	```
		glob
		matplotlib
		numpy
		opencv
		pandas
		PIL
		tqdm
		torch
		torchvision
	```
### Datasets

The datasets can be downloaded by uncommenting and running the first cell in the main jupyter notebook.