import numpy as np
from scipy import signal
from skimage import color, filters, measure, transform
import cnn

k1 = np.array([0.5,0,-0.5]).reshape(3,1)
k2 = k1.T
k3 = np.array([0,0.5,-0.5,0]).reshape(2,2)
k4 = k3.T

def add_gs(img): #add a grayscale representation as a channel, used for convolutions
    z = np.zeros(shape = (img.shape[0],img.shape[1],img.shape[2]+1))
    z[:,:,0:img.shape[2]] = img
    z[:,:,img.shape[2]] = color.rgb2gray(img)
    return z

def add_convolve(img,n,k,sigma = 0.1): #convolve a given channel and add the output as another channel
    z = np.zeros(shape = (img.shape[0],img.shape[1],img.shape[2]+1))
    img_smooth = filters.gaussian(img[:,:,n],sigma)
    z[:,:,0:img.shape[2]] = img
    z[:,:,img.shape[2]] = signal.convolve2d(img_smooth,k,mode='same')
    return z

def convolve(img,n,k): #convolve a given channel
    img_copy = img.copy()
    img_copy[:,:,n] = signal.convolve2d(img_copy[:,:,n],k,mode='same')
    return img_copy

def pooling(img,n,m=None): #max pooling
    if not m:
        m = n
    z = np.zeros(shape = (int(np.ceil(img.shape[0]/n)),int(np.ceil(img.shape[1]/m)),img.shape[2]))
    for i in range(img.shape[2]):
        z[:,:,i] = measure.block_reduce(img[:,:,i],(n,m),np.max)
    return z

def data_augmentation(img): #Simple transformations to augment data, used when more pictures than possible requested for training
    case = np.random.randint(0,4)
    img_copy = img.copy()
    #case 0 as is
    if case == 1:
        #vertical flip
        img_copy[:,:] = img_copy[:,::-1]
    if case == 2:
        #horizontal flip
        img_copy[:,:] = img_copy[::-1,:]
    if case == 3:
        #vertical and horizontal flip
        img_copy[:,:] = img_copy[::-1,::-1]
        
    random_degree = np.random.uniform(-30, 30)
    img_copy = transform.rotate(img_copy, random_degree)
    
    return img_copy

def transform_img(img): #Obtain hopefully more meaningful data from the image and pool the info to reduce number of inputs
    img_copy = img.copy()
    img_copy = add_gs(img_copy)
    img_copy = add_convolve(img_copy,3,k1)
    img_copy = add_convolve(img_copy,3,k2)
    img_copy = add_convolve(img_copy,3,k3)
    img_copy = add_convolve(img_copy,3,k4)
    img_copy = cnn.relu(img_copy)
    img_copy = pooling(img_copy,2,2)
    img_copy = add_convolve(img_copy,4,k2)
    img_copy = add_convolve(img_copy,5,k1)
    img_copy = add_convolve(img_copy,6,k4)
    img_copy = add_convolve(img_copy,7,k3)
    img_copy = cnn.relu(img_copy)
    img_copy = pooling(img_copy,2,2)
    
    return img_copy
