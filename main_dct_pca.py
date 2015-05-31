import cv2
import numpy as np
import random
import cPickle as pickle
import os
import sys
from django import template
from scipy.linalg import eigh
import scipy
from numpy import *
from pylab import *
from cmath import *
import imagehash
from PIL import Image
from skimage.measure import structural_similarity as ssim

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

FOLDER = "data"
LIMIT = 423#1459 #limits the number of images to be considered as part of the database
SIZE = 10 #size of reshaped faces
count_useless_images = 0
TOTAL_NO_IMAGES = 100
NO_SIMILAR = 10
MEMORY = 'memory_file.dat'
RESULTS = 'results'
K = 1

#List containing names of the images
NAME = range(1, LIMIT+1)
NAME = [str(i) for i in NAME]

#Cascade for the face
CASC_PATH = 'lbpcascades/lbpcascade_frontalface.xml'#'haarcascades/haarcascade_frontalface_alt.xml'
FACE_CASCADE = cv2.CascadeClassifier(CASC_PATH)

def preprocess(image, name):

    """Converts the image to gray scale and also resizes the image"""
    
    global SIZE, count_useless_images
    size = 50
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
#        print name, "rgb prob"
        image = None
        count_useless_images += 1
        return image
        
    faces = FACE_CASCADE.detectMultiScale(image)
#    print name, faces
    if ((len(faces) > 1) or (len(faces) == 0)):
        image = None
        count_useless_images += 1
#        if len(faces) > 1:
##            print name, "many faces"
#        else:
##            print name, "no face"
        return image
    
    for (x,y,w,h) in faces:
        image = image[y:y+h, x:x+w]
    
    if len(image) == 0 or image == []:
        count_useless_images += 1
        image = None
    else:
        try:
#            image = cv2.resize(image, (SIZE, SIZE))
            image = cv2.resize(image, (size, size))
        except:
#            print image, "resize bullshit, is the type of the image"
            image = None
            
#    if image is not None:
#        cv2.imshow("face", image)
#        cv2.waitKey(0)
#    if name == '51':
#        cv2.imwrite("fiftyone.jpg", image)
#    elif name == '99':
#        cv2.imwrite("nintynine.jpg", image)
#        print "done"
    return image

def transform(image):
    
    """Some normalization is applied on the image and then the image is converted into a row vector so that it can be dealt with easily"""
    
#    image = image / 255.0 #For now, we apply a very naive method of normalization
    image = dct(image)
#    image = cv2.resize(image, (SIZE, SIZE))
    image = image.ravel()
#    image -= image.mean()
    return image

def init():

    """Load the images and the dataset randomly so as to not hamper training"""    
    
    data = np.zeros(shape=(LIMIT, SIZE*SIZE))
    images = []
#    k = 0
    for itr in range(LIMIT):

        img = cv2.imread(FOLDER + "/" + NAME[itr] + ".jpg")
        img = preprocess(img, NAME[itr])
        if img is None:
            continue
        images.append(img)
#        print img[50,50]
#        cv2.imshow("face", img)
#        cv2.waitKey(0)
        data[itr] = transform(img)#img.ravel() #normalize_and_flatten(img)
#        k += 1
        data = np.array(data).astype('float32')
    return [data, images]

def dct(img):
    
    """Computes discrete cosine transform of an image"""
    
    global SIZE
    imf = np.float32(img)/255.0  # float conversion/scale
    dst = cv2.dct(imf)           # the dct
    img = np.uint8(dst)*255.0    # convert back
    img = img[0:SIZE,0:SIZE]
    return img

def run_pca(data):
    
    """Does PCA on the face dataset and return the transformed faces"""
    
    global LIMIT, SIZE, K
    
#    An array of zeroes to hold normalised data
    data_normalised = np.zeros(shape=(LIMIT, SIZE*SIZE))
    data_normalised = np.array(data_normalised).astype('float32')
    
    mean_vector = data.mean(axis=0) #mean of all columns
    std_vector = data.std(axis=0)
    
    for i in range(SIZE*SIZE):
        if float(std_vector[i]) == 0:
            std_vector[i] = 1.0
            
#    m,n = data.shape
    #feature normalising the data, here the features have a mean of 0 and a std of 1
    for i in range(LIMIT): 
        data_normalised[i] = (data[i] - mean_vector) / std_vector
    
#    Findig covariance matrix
    covariance_matrix = np.dot(data_normalised.T, data_normalised) / LIMIT     
    
    print "covariance done"
    U, S, V = np.linalg.svd(covariance_matrix)
    print "svd done"
#    Knowledgibly finding the number of eigen vectors
    k = 0
    req_variance = .99 #retain 99% variance
    current_sum = 0.0
    total_sum = sum(S)
    #    Find k
    for i in range(len(S)):
        current_sum += S[i]
        var = (current_sum / total_sum)
    #    print var
        if (var > req_variance):
            k = i + 1
            print "k =", k, "var = ", var
            break
    K = k
    print "about to display"
    U_reduce = U[:,0:k] #Selecting only top k eigen vectors
#    Displaying the first eigen face
    eigen_face = U_reduce[:,0].T
    eigen_face = (eigen_face * std_vector) + mean_vector
    print eigen_face.shape
    to_show = eigen_face.reshape(SIZE,SIZE)

#   Display image
#    figure()
#    imshow(to_show)
#    title('PCs # ')
#    gray()
#    show()

    transformed_data = np.dot(data_normalised, U_reduce)
    return [transformed_data, mean_vector, std_vector, U_reduce]

def pca_single_image(image_name, mean_vector, std_vector, U_reduce):

    """Function to perform pca on a single image"""
    
    image = cv2.imread(image_name + ".jpg")
    image = preprocess(image, image_name)
    if image is None:
        print "Couldn't load image"
        return False
    image_array = transform(image)#image.ravel()
#    print "charlie image", image_array
    normalised_image_array = (image_array - mean_vector) / std_vector
    transformed_image_array = np.dot(normalised_image_array, U_reduce)
    return transformed_image_array
    

def distance(x, y):
    
    """Find the mean absolute error between two column vectors x and y and add z to this. Here z contains the values learnt from the user"""
#    global K
#    X = zeros((10, K))
#    Y = zeros((10, K))
#    X[0] = x
#    Y[0] = y
#    return ssim(X, Y)
#    return sum((x-y)**2)
    return sum(abs(np.subtract(x, y)))
    
    
def compare_images(image, data):

    """Finds the best match"""
    
    global LIMIT

    #Finds the least distance between test image and the dataset
    pos = range(LIMIT - 1) # contains the indices of all the images in the database
    value = [] # will contain the mean square errors later

    for i in range(len(pos)):
        value.append(distance(image, data[i]))
        if i == 0:
            continue
        k = i
        while((value[k] < value[k - 1]) and k > 0):
            #swap positions
            pos[k], pos[k - 1] = pos[k - 1], pos[k]
            #swap values
            value[k], value[k - 1] = value[k - 1], value[k]
            k -= 1
    print
    #Now pos will contain the indexes to images in ascending order of distance
    return pos, value


#def store_data(data):
#    
#    pickle.dump(data, open(MEMORY, 'wb'))
    
def render_images(pos, test_img_path):
    
    html = """<html>
        <head><title>Similar images</title></head>

        <body>
        <h1 align="center">Test image</h1>
        <p align="center"><img src="{{ test_image }}" alt="test image"></p>

        <h1 align="center">Similar images</h1>

        <ol>
        {% for item in list %}
            <li><img src="{{ item }}" alt="similar image" height="100" width="100"></li>
            </br>
        {% endfor %}
        <ol>

        </body>
        </html>
        """    
    t = template.Template(html)
    pos = [FOLDER + "/" + str(i + 1) + ".jpg" for i in pos]
    c = template.Context({'list': pos, 'test_image': test_img_path})
    html = t.render(c)
    f = open(RESULTS + ".html", 'w')
    f.write(html)


def main():

    global NO_SIMILAR
    [data, images] = init()
#    for i in range(5):
#        a = data[i,:].reshape(SIZE,SIZE)
#        a = array(a)
##        a[0:2,:] = 255
#        print a.shape
#        cv2.imshow("a", a)
#        cv2.waitKey(0)
#        cv2.imwrite("a.jpg", a)
#        a = Image.open("a.jpg")
#        h = str(imagehash.dhash(a))
#        print h

#    print "charlie", data[60,:]
    [transformed_data, mean_vector, std_vector, U_reduce] = run_pca(data)
#    print "TD\n", transformed_data[:,0:5]
#    print "charlie transforemed main", transformed_data[60,:]
#    image_name = FOLDER + "/243"
    image_name = FOLDER + "/352"
#    image_name = FOLDER + "/297"
#    image_name = FOLDER + "/304"

    transformed_image = pca_single_image(image_name, mean_vector, std_vector, U_reduce)
#    print "TI\n", transformed_image
#    print "charlie transforemed", transformed_image
    pos, value = compare_images(transformed_image, transformed_data)
    pos = array(pos)
    pos += 1
    print "count", count_useless_images
    print "pos\n", pos[0:100]
    print "value \n", value[0:100]
    pos = pos[0:NO_SIMILAR]
    
#    pos += 1
#    print pos
    test_img_path = image_name + ".jpg"
#    To create a html page
    render_images(pos, test_img_path)
    
    for i in range(NO_SIMILAR):
        figure()
        my_image = imread(FOLDER + "/" + str(int(pos[i])) + ".jpg") 
        imshow(flipud(my_image))
        title(str(i + 1))
    show()
main()
