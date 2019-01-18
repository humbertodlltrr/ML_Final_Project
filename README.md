# ML Final Project
The project is a simple implementation of a convolutional neural network. Images are transformed before being utilized as inputs for the ConvNet. The neural network consists of one hidden layer. 

There are four python files. These files run on Python 3, 3.6.5, with Anaconda distribution. Program runs without issues in the Spyder IDE. Otherwise just be sure to be able to make the following imports: numpy, pickle, os, tarfile, urllib, maplotlib, scipy and skimage. Running tests should download if neccessary the cifar10 data set from the following url https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz. If problems arise simply having the cifar-10-python.tar.gz file in the same directory as the python files should work. 

Tests can be done in test.py and an example of the available functions are shown and commented accordingly. Regardless ahead I put some notes on the most important bits in test.py.

# 1.Firstly it is required to indicate which classes of the data set are to be used.

#0:airplane, 1:automobile, 2:bird, 3:cat, 4:deer, 5:dog, 6:frog, 7:horse, 8:ship, 9:truck
#Insert into the array the classes you want to include, below example for comparing airplanes and automobiles
#All 10 classes can be used at once
cls = np.array([0,1])

# 2. Before starting there are some parameters that can be altered to obtain worse/better results

#Some parameters to play with, more info on the following comment
epochs = 10
batch_size = int(total_tr_images/epochs)
no_hidden_layers = 64
wmin = -0.4
wmax = 0.4
alpha = 0.001
eta = 0.01

# 3. train_cnn is where most of the machine learning is going on. Takes the training and test values to train the neural network and record how well are the predictions going. 

"""
The following trains the neural network and also returns data it collects while training.
Inputs: Xtr - training images
        Ytr - training labels
        Xts - test images
        Yts - test labels
        epochs - number of epochs
        batch_size - images trained on per epoch
        no_hidden_layers - number of hidden layers
        cls.size - number of classes
        (wmin,wmax) - Initial weight values distribution range 
        alpha - momentum
        eta - learning rate
Outputs: w1 - last first set of weights
         w2 - last second set of weights
         total_cost - cost obtained as each image was fed forward
         pred_acc - accuracy after a given epoch
         y_pred - guesses for Xts
"""
w1,w2,total_cost,pred_acc,y_pred = cnn.train_cnn(Xtr,Ytr,Xts,Yts,epochs,batch_size,no_hidden_layers,cls.size,(wmin,wmax),alpha,eta)

# 4. Simple function to visualize the cost and accuracy as plots

#Input a 1D array to plot it, good for the cost and accuracy
print("Cost function")
cnn.visualize_plot(total_cost)
print("CNN accuracy over epochs")
cnn.visualize_plot(pred_acc)

# 5. Simple function to visualize a set of n either correctly or incorrectly labeled images

#Visualize a set of images, True -> correct predictions, False -> incorrect predictions, last value is number of pictures
cnn.visualize_images(classes,Xts,Yts,y_pred.flatten(),True,5)
cnn.visualize_images(classes,Xts,Yts,y_pred.flatten(),False,5)

# Other python files
cifar10.py deals with the data set, downloading/extracting/reading
imgs.py processes the images before utilizing them as inputs, convolution/pooling/relu
cnn.py holds most of the functions pertaining to machine learning
