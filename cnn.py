import numpy as np
import imgs
import matplotlib.pyplot as plt

no_variables_input = 8*8*12

def one_hot_encoding(cls,no_classes): #change labels from an integer to a vector respresentation
    cls_encoded = np.zeros((no_classes,cls.size))
    for i in range(cls.size):
        cls_encoded[int(cls[i])][i] = 1
    return cls_encoded

def relu(z): #Simple max(0,z)
    z[z<0] = 0
    return z

def gradient_relu(z): #0 where 0, 1 elsewhere
    return np.where(z > 0, 1, 0)

def softmax(z): #In order to use cross entropy one softmax activation is used to avoid log(0)
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    return sm

def cost_function(out, cls_encoded): #Cross entropy
    return -np.sum(L*np.log(S))

#Simple bias to reduce overfitting
def column_bias(z):
    return np.append(np.ones(z.T.shape[1]),z.T).reshape(z.T.shape[0]+1,z.T.shape[1]).T
def row_bias(z):
    return np.append(np.ones(z.shape[1]),z).reshape(z.shape[0]+1,z.shape[1])
def add_bias(z,b):
    if b:
        return column_bias(z)
    else:
        return row_bias(z)

#First set of weights
def first_weights(n_features, n_hidden,no_classes,dist):
    w1 = np.random.uniform(dist[0], dist[1], size=n_hidden*(n_features+1))
    w1 = w1.reshape(n_hidden, n_features+1)
    w2 = np.random.uniform(dist[0], dist[1], size=no_classes*(n_hidden+1))
    w2 = w2.reshape(no_classes, n_hidden+1)
    
    return w1, w2

def feed_forward(x, w1, w2): #Feed forward an image into the net
    x_copy = x.copy().reshape(1,no_variables_input)
    a1 = add_bias(x_copy, True)
    
    z2 = w1.dot(a1.T)
    a2 = relu(z2)
    a2 = add_bias(a2, False)
    z3 = w2.dot(a2)
    a3 = softmax(z3)
    return a1, z2, a2, z3, a3

def predict(x, w1, w2): #To obtain y_pred given a set of weights give labels to and image set x
    y_pred = []
    for img in x:
        inp = imgs.transform_img(img)
        a1, z2, a2, z3, a3= feed_forward(inp, w1, w2)
        y_pred.append(np.argmax(a3, axis=0))
    return y_pred

def calc_grad(a1, a2, a3, z2, z3, y_enc, w1, w2): #gradient descent
    delta3 = a3 - y_enc
    z2 = add_bias(z2,False)
    delta2 = w2.T.dot(delta3)*gradient_relu(z2)
    delta2 = delta2[1:,:]
    
    grad1 = delta2.dot(a1)
    grad2 = delta3.dot(a2.T)

    return grad1, grad2


def cnn_acc(w1,w2,imgTs,clsTs): #Takes a pair of weights and returns succes rate of guesses on images compared to their labels
    correct = 0
    for i in range(imgTs.shape[0]):
        img = imgs.transform_img(imgTs[i])
        a1, z2, a2, z3, a3 = feed_forward(img, w1, w2)
        if clsTs[i] == np.argmax(a3):
            correct +=1
    return correct/imgTs.shape[0]*100

def train_cnn(imgTr,clsTr,imgTs,clsTs,epochs,batches,no_hidden_layers,no_classes,dist,a,e): #Main function, trains the cnn and returns some other details
    clsTr_enc = one_hot_encoding(clsTr,no_classes)
    epochs = epochs
    batch = batches
    
    rep = False
    if batch * epochs > clsTr.shape[0]:
        rep = True
    
    train_sequence = np.random.choice(np.arange(clsTr.shape[0]),batch*epochs,replace=rep)
    train_batches = np.array_split(train_sequence,epochs)
    
    w1,w2 = first_weights(no_variables_input,no_hidden_layers,no_classes,dist)
    
    #learning parameters
    alpha = a
    eta = e
    dec = 0.00001
    
    delta_w1_prev = np.zeros(w1.shape)
    delta_w2_prev = np.zeros(w2.shape)
    total_cost = []
    pred_acc = []
    
    for i in range(epochs):
        eta /= (1 + dec*i)
        for j in train_batches[i]:
            img = imgTr[j].copy()
            if rep:
                img = imgs.data_augmentation(img)
            img = imgs.transform_img(img)
            a1, z2, a2, z3, a3 = feed_forward(img, w1, w2)
            cost = cost_function(a3,clsTr_enc[:,j].reshape(1,no_classes))
            total_cost.append(cost)
            
            #backpropagation
            grad1,grad2 = calc_grad(a1, a2, a3, z2, z3, clsTr_enc[:,j].reshape(no_classes,1), w1, w2)
            delta_w1, delta_w2 = eta * grad1, eta * grad2
            w1 -= delta_w1 + alpha * delta_w1_prev
            w2 -= delta_w2 + alpha * delta_w2_prev
            
            delta_w1_prev, delta_w2_prev = delta_w1, delta_w2    
        print("End epoch #",i+1)
        #test_sequence = np.random.choice(np.arange(clsTs.shape[0]),500,replace=False)
        pred_acc.append(cnn_acc(w1,w2,imgTs,clsTs))

    return w1,w2,total_cost,pred_acc, np.array(predict(imgTs,w1,w2))

def visualize_plot(y): #plot 1D array
    x = [i for i in range(len(y))]
    plt.figure()
    plt.plot(x,y)
    plt.show()


def visualize_images(classes,test_x,test_y,y_pred,correct=True,n=1): #visualize a set of images either correctly guessed or not
    
    if correct:
        miscl_img = test_x[test_y == y_pred]
        correct_lab = test_y[test_y == y_pred]
        miscl_lab = y_pred[test_y == y_pred] 
    else:
        miscl_img = test_x[test_y != y_pred]
        correct_lab = test_y[test_y != y_pred]
        miscl_lab = y_pred[test_y != y_pred]
    
    sequence = np.random.choice(np.arange(miscl_img.shape[0]),n,replace=False)
    
    for i in sequence:
        plt.figure()
        plt.imshow(miscl_img[i])
        plt.show()
        print("Label: ",classes[correct_lab[i]])
        print("Guess: ",classes[miscl_lab[i]])

