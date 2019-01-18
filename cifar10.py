import numpy as np
import pickle
import os
import tarfile
import urllib.request
import matplotlib.pyplot as plt

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
cifar10_targz_filename = "cifar-10-python.tar.gz" #compressed CIFAR-10 file
cifar10_extract_dirname = "CIFAR-10" #name of directory for extracted CIFAR-10 data
batch_filepath = os.path.join(cifar10_extract_dirname,"cifar-10-batches-py/")

tr_files = 5 #training files, can be a value between 1 and 5
image_dimensions = [32,32,3] #length, width, channels


def cifar10_targz_extract(file_name): #Assuming extracting from a .tar.gz file
    if os.path.isfile(file_name): #If .tar.gz already in the same directory
        tarfile.open(name=file_name, mode="r:gz").extractall(cifar10_extract_dirname) #Extract contents of tar.gz
    else: #Download .tar.gz
        print("Downloading cifar-10-python.tar.gz")
        file_path, _ = urllib.request.urlretrieve(url,file_name) #Download dataset
        print("Finished download")
        tarfile.open(name=file_name, mode="r:gz").extractall(cifar10_extract_dirname) #Extract contents of tar.gz
        

def load_batch(file_name):
    data = pickle.load(open(os.path.join(batch_filepath,file_name),mode ='rb'),encoding='bytes') #Unpickler 
    
    images = np.array(data[b'data']) / 255.0 #Change images from 0-255 values to 0-1
    classes = np.array(data[b'labels']) #Image labels

    return images, classes #Return images and classes

def load_class_batch():
    data = pickle.load(open(os.path.join(batch_filepath,"batches.meta"),mode ='rb'),encoding='bytes') #Unpickler 
    
    classes = [i.decode("utf-8") for i in data[b'label_names']]
    
    return classes #Return images and classes
    
     
def get_training_data():
    images = np.array([]) #Empty array to where images will be added
    classes = np.array([]) #Empty array to where labels will be added
    
    batches = ["data_batch_"+str(i+1) for i in range(tr_files)] #batches named data_batch_1,data_batch_2,.... up to 5
    for b in batches: #For each batch
        ibatch, cbatch = load_batch(b) #load the data
        images = np.append(images,ibatch) #Get image data, at this point a single 1D array of values
        classes = np.append(classes,cbatch) #Get label data
    
    images = images.reshape([-1, image_dimensions[2], image_dimensions[0], image_dimensions[1]]) #reshape the 1D array into images 
    images = images.transpose([0, 2, 3, 1]) #rearrange values

    return images, classes

def get_test_data():
    images, classes = load_batch("test_batch")
    images = images.reshape([-1, image_dimensions[2], image_dimensions[0], image_dimensions[1]]) #reshape the 1D array into images 
    images = images.transpose([0, 2, 3, 1]) #rearrange values
    
    return images, classes

def get_data():
    cifar10_targz_extract(cifar10_targz_filename) #Extract from tar.gz, download if required
    imgTr, clsTr = get_training_data() #Get training data
    imgTs, clsTs = get_test_data() #Get test data
    classes = load_class_batch()
    
    return imgTr, clsTr, imgTs, clsTs, classes

def show_images(img_array, label_array,label): #Given an image array, label array and label
    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(16): #Show the first 16 images
        img = img_array[label_array==label][i]
        ax[i].imshow(img)
    plt.show()
