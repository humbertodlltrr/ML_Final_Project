# ML Final Project
The project is a simple implementation of a convolutional neural network. Images are transformed before being utilized as inputs for the ConvNet. The neural network consists of one hidden layer. 

There are four python files. These files run on Python 3, 3.6.5, with Anaconda distribution. Program runs without issues with the Spyder IDE in the Anaconda Navigator. Otherwise I believe being able to make the following imports should suffice: numpy, pickle, os, tarfile, urllib, maplotlib, scipy and skimage. cifar10.py get_data() function should download if neccessary the cifar10 data set from the following url https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz. If problems arise simply having the cifar-10-python.tar.gz file in the same directory as the python files should work. This function is already included in the python file test.py where tests are intended to be performed.

Tests can be done in test.py and an example of the available functions are shown and commented accordingly. Regardless ahead I put some notes on the most important bits in test.py.

# 1.Firstly it is required to indicate which classes of the data set are to be used.

#0:airplane, 1:automobile, 2:bird, 3:cat, 4:deer, 5:dog, 6:frog, 7:horse, 8:ship, 9:truck <br>
#Insert into the array the classes you want to include, below example for comparing airplanes and automobiles  <br>
#All 10 classes can be used at once 

cls = np.array([0,1])

# 2. Before starting there are some parameters that can be altered to obtain worse/better results

#Some parameters to play with, more info on the following comment <br>
epochs = 10 <br>
batch_size = int(total_tr_images/epochs) <br>
no_hidden_layers = 64 <br>
wmin = -0.4 <br>
wmax = 0.4 <br>
alpha = 0.001 <br>
eta = 0.01

# 3. train_cnn is where most of the machine learning is going on. Takes the training and test values to train the neural network and record how well are the predictions going. 

""" <br>
The following trains the neural network and also returns data it collects while training. <br>
Inputs: Xtr - training images <br>
        Ytr - training labels <br>
        Xts - test images <br>
        Yts - test labels <br>
        epochs - number of epochs <br>
        batch_size - images trained on per epoch <br>
        no_hidden_layers - number of hidden layers <br>
        cls.size - number of classes <br>
        (wmin,wmax) - Initial weight values distribution range  <br>
        alpha - momentum <br>
        eta - learning rate <br>
Outputs: w1 - last first set of weights <br>
         w2 - last second set of weights <br>
         total_cost - cost obtained as each image was fed forward <br>
         pred_acc - accuracy after a given epoch <br>
         y_pred - guesses for Xts <br>
""" <br>
w1,w2,total_cost,pred_acc,y_pred = cnn.train_cnn(Xtr,Ytr,Xts,Yts,epochs,batch_size,no_hidden_layers,cls.size,(wmin,wmax),alpha,eta)

# 4. Simple function to visualize the cost and accuracy as plots

#Input a 1D array to plot it, good for the cost and accuracy <br>
print("Cost function") <br>
cnn.visualize_plot(total_cost) <br>
print("CNN accuracy over epochs") <br>
cnn.visualize_plot(pred_acc)

# 5. Simple function to visualize a set of n either correctly or incorrectly labeled images

#Visualize a set of images, True -> correct predictions, False -> incorrect predictions, last value is number of pictures <br>
cnn.visualize_images(classes,Xts,Yts,y_pred.flatten(),True,5) <br>
cnn.visualize_images(classes,Xts,Yts,y_pred.flatten(),False,5)

# Other python files
<b>cifar10.py</b> deals with the data set, downloading/extracting/reading <br>
<b>imgs.py</b> processes the images before utilizing them as inputs, convolution/pooling/relu <br>
<b>cnn.py</b> holds most of the functions pertaining to machine learning
