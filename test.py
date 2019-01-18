import cnn
import cifar10
import numpy as np

#Read the CIFAR10 data
imgTr,clsTr,imgTs,clsTs,classes = cifar10.get_data()

#0:airplane, 1:automobile, 2:bird, 3:cat, 4:deer, 5:dog, 6:frog, 7:horse, 8:ship, 9:truck
#Insert into the array the classes you want to include, below example for comparing cats and dogs
#All 10 classes can be used at ones with poor results given a naive implementation
cls = np.array([0,1])

#Sort the data out based on the previous array
cls.sort(axis = 0)
a = np.where(np.isin(clsTr,cls))
b = np.where(np.isin(clsTs,cls))
Xtr = imgTr[a]
Ytr = clsTr[a]
Xts = imgTs[b]
Yts = clsTs[b]
for i in range(cls.size):
    Ytr[Ytr == cls[i]] = i
    Yts[Yts == cls[i]] = i
classes = np.array(classes)[cls]
total_tr_images = Xtr.shape[0]

#Some parameters to play with, more info on the following comment
epochs = 10
batch_size = int(total_tr_images/epochs)
no_hidden_layers = 64
wmin = -0.4
wmax = 0.4
alpha = 0.001
eta = 0.01

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
#Accuracy on test set after all epochs
print("Final accuracy: ",cnn.cnn_acc(w1,w2,Xts,Yts),"%")

#Input a 1D array to plot it, good for the cost and accuracy
print("Cost function")
cnn.visualize_plot(total_cost)
print("CNN accuracy over epochs")
cnn.visualize_plot(pred_acc)

#Visualize a set of images, True -> correct predictions, False -> incorrect predictions, last value is number of pictures
cnn.visualize_images(classes,Xts,Yts,y_pred.flatten(),True,5)
cnn.visualize_images(classes,Xts,Yts,y_pred.flatten(),False,5)

