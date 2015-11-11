import cv2
import numpy as np
import random
import cPickle as pickle
import os
import sys
from django import template
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LinearRegression
from sklearn import svm
from itertools import izip
import math
import scipy
from skimage.filters import sobel#, roberts,  scharr
from scipy.stats import itemfreq 
#from scipy.linalg import eigh
#import scipy
#from pylab import *
#from cmath import *
#import imagehash
#from PIL import Image
#from skimage.measure import structural_similarity as ssim

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

FOLDER = "data"
#test_image_name = FOLDER + "/244" #Name or path of test image -> Omit .jpg(only .jpg)
LIMIT = 423#1459 #limits the number of images to be considered as part of the dataBASE
SIZE = 50#10#20#10 #size of reshaped faces
count_useless_images = 0
#TOTAL_NO_IMAGES = 100
NO_SIMILAR = 10
WINDOW = 10
FEATURES_LENGTH = ((SIZE / WINDOW) ** 2) * 59
FACTOR_REINFORCEMENT_FEATURES = 4

MEMORY_DATA = 'memory_file.dat'
MEMORY_REINFORCE = 'memory_reinforce.dat'
RESULTS = 'results'
K = 1
NO_TREES = 300
additional = 30
extra_removal = 0.1
#List containing names of the images
NAME = range(1, LIMIT+1)
NAME = [str(i) for i in NAME]

#Cascade for the face
CASC_PATH = 'lbpcascades/lbpcascade_frontalface.xml'#'haarcascades/haarcascade_frontalface_alt.xml'
FACE_CASCADE = cv2.CascadeClassifier(CASC_PATH)
lefteye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_lefteye_2splits.xml')
righteye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_righteye_2splits.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_eyepair_small.xml')

UNIFORM_PATTERNS = np.array([0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255])
BASE = np.array(range(0,59))
SIZE = 50
#human_features = []


def my_lbp(X):

    """Return the  LBP of an image"""
    
    X = np.asarray(X)
    X = (1<<7) * (X[0:-2,0:-2] >= X[1:-1,1:-1]) \
        + (1<<6) * (X[0:-2,1:-1] >= X[1:-1,1:-1]) \
        + (1<<5) * (X[0:-2,2:] >= X[1:-1,1:-1]) \
        + (1<<4) * (X[1:-1,2:] >= X[1:-1,1:-1]) \
        + (1<<3) * (X[2:,2:] >= X[1:-1,1:-1]) \
        + (1<<2) * (X[2:,1:-1] >= X[1:-1,1:-1]) \
        + (1<<1) * (X[2:,:-2] >= X[1:-1,1:-1]) \
        + (1<<0) * (X[1:-1,:-2] >= X[1:-1,1:-1])
    return X

def preprocess(image, name):

    """Converts the image to gray scale and also resizes the image"""
    
    global SIZE, count_useless_images, additional, lefteye_cascade, righteye_cascade, eye_cascade
    size = 50
    extra_removal = 0.1 
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
        
        img = image[y:y+h,x:x+w]#[int(y+h*extra_removal):int(y+(1-extra_removal)*h), int(x+w*extra_removal):int(x+(1-extra_removal)*w)]
#        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite("data_face/" + name + ".jpg", image)
        image = img
        image = cv2.resize(image, (SIZE, SIZE))
        cv2.imwrite("data_face/" + name + ".jpg", image)
    image = img
    if w != h:
        print "not equal" 
    generic_size = 400
    
    
    after_sobel_size = 500
    if len(image) == 0 or image == []:
        count_useless_images += 1
        image = None
    else:
        try:
            length, width = image.shape
            eyes = eyes_cascade.detectMultiScale(image)
            if len(eyes) == 1:
#                print "yess1"
                for (ex,ey,ew,eh) in eyes:
                    X1, Y1, X2, Y2 = ex, ey, ex+ew, ey+eh
        #            print X1, X2
#                    cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                flag = 1
            
            if not flag:
                eyes = lefteye_cascade.detectMultiScale(image)
                if len(eyes) == 2:
#                    print "yess2"
                    x1,y1,w1,h1 = eyes[0]
                    x2,y2,w2,h2 = eyes[1]
                    X1 = min(x1, x2)
                    X2 = max(x1+w1, x2+w2)
                    Y1 = min(y1, y2)
                    Y2 = max(y1+h1, y2+h2)
            #            for (ex,ey,ew,eh) in eyes:
#                    cv2.rectangle(image,(X1,Y1),(X2,Y2),(0,255,0),2)
#                    for (ex,ey,ew,eh) in eyes:
#                        cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        #            print X1, X2
                    flag = 1

            if not flag:
                eyes = righteye_cascade.detectMultiScale(image)
                if len(eyes) == 2:
#                    print "yess3"
                    x1,y1,w1,h1 = eyes[0]
                    x2,y2,w2,h2 = eyes[1]
                    X1 = min(x1, x2)
                    X2 = max(x1+w1, x2+w2)
                    Y1 = min(y1, y2)
                    Y2 = max(y1+h1, y2+h2)
            #            for (ex,ey,ew,eh) in eyes:
#                    cv2.rectangle(image,(X1,Y1),(X2,Y2),(0,255,0),2)
#                    for (ex,ey,ew,eh) in eyes:
#                        cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    flag = 1
        #            print X1, X2    
        #    if i == 26:
        #    cv2.imshow("image", image)
        #    cv2.waitKey(0)
            cv2.imwrite("data_face/" + name + ".jpg", image)
            if not flag:
                image = image[int(width*extra_removal):int((1-extra_removal)*width), int(length*extra_removal):int((1-extra_removal)*length)]
            
            else:
        #        try:
                y1 = int(Y1 - 0.3*width)
                if y1 < 0:
                    y1 = 0
                y2 = int(Y2 + 0.5*width)
                if y2 > width:
                    y2 = width
                image = image[y1:y2,X1:X2]
#        if (True):
#            image = cv2.resize(image, (after_sobel_size,after_sobel_size))
#            
#            kernel = np.ones((5,5),np.float32)/25
#            image = cv2.filter2D(image,-1,kernel)#remove gaussian noise
#            
#            dst = sobel(image) #sobel filtering
#            dst[dst > np.mean(dst)] = 1#Thresholding
#            dst[dst < np.mean(dst)] = 0

##            finding coordinates of the face
#            for j in range(after_sobel_size):
#                if sum(dst[j, :]) > 0:
#                    x1 = j
#                    break
#            for j in range(after_sobel_size):
#                if sum(dst[:, j]) > 0:
#                    y1 = j
#                    break
#            for j in reversed(range(after_sobel_size)):
#                if sum(dst[j, :]) > 0:
#                    x2 = j
#                    break
#            for j in reversed(range(after_sobel_size)):
#                if sum(dst[:, j]) > 0:
#                    y2 = j
#                    break
##            print x1, y1, x2, y2
#            image = image[x1:x2, y1:y2]

##            Remove ears
#            image = cv2.resize(image, (after_sobel_size, after_sobel_size))
#            image = image[additional:after_sobel_size-additional, additional:after_sobel_size-additional]
            
#            image = cv2.resize(image, (size, size))
            image = cv2.resize(image, (SIZE, SIZE))
            cv2.imwrite("data_edge/" + name + ".jpg", image)
            
            
#        else:
        except:
#            print image, "resize bullshit, is the type of the image"
            image = image[int(width*extra_removal):int((1-extra_removal)*width), int(length*extra_removal):int((1-extra_removal)*length)]
#            image = cv2.resize(image, (size, size))
            image = cv2.resize(image, (SIZE, SIZE))
            cv2.imwrite("data_edge/" + name + ".jpg", image)
#            image = None
            
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
#    image = cv2.equalizeHist(image)
#    image = dct(image)
#    image = lbp(image)
#    image = cv2.resize(image, (SIZE, SIZE))
#    image = image.ravel()
#    image -= image.mean()
    return lbp(image)
#    return image.ravel()
    
def init():

    """Load the images and the dataset randomly so as to not hamper training"""    
    
    data = np.zeros(shape=(LIMIT, FEATURES_LENGTH))# SIZE*SIZE))
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

def lbp(im_gray):

    """Returns LBP histogram of an image"""
    
    global SIZE, WINDOW, UNIFORM_PATTERNS, BASE
    im_gray = cv2.resize(im_gray, (SIZE, SIZE))
    lbp_hist = np.array([])
    for i1 in range(0,SIZE,WINDOW):
        for j1 in range(0,SIZE,WINDOW):
            box = im_gray[j1:j1+WINDOW, i1:i1+WINDOW]
#            figure()
#            imshow(box)
#            gray()
#            show()
            lbp = my_lbp(box)
#            print lbp.shape
            lbp = lbp.ravel()
            map_array = np.zeros((lbp.shape[0] + 1))
            i = 0
            for x in np.nditer(lbp):
                try:
                    map_array[i] = np.where(UNIFORM_PATTERNS==x)[0][0]            
                except:
                    map_array[i] = 58
                i += 1
            map_array = np.concatenate((map_array, BASE))
            x = itemfreq(map_array)
            hist = np.array(x[:,1] - 1).astype('int')
        #    print x
        #    print type(x)
        #    print sum(hist)
#            print hist
            lbp_hist = np.concatenate((lbp_hist, hist))
    return lbp_hist
    

def run_pca(data):
    
    """Does PCA on the face dataset and return the transformed faces"""
    
    global LIMIT, SIZE, K
    
#    An array of zeroes to hold normalised data
    data_normalised = np.zeros(shape=(LIMIT, FEATURES_LENGTH))# SIZE*SIZE))
    data_normalised = np.array(data_normalised).astype('float32')
    
    mean_vector = data.mean(axis=0) #mean of all columns
    std_vector = data.std(axis=0)
    
    for i in range(FEATURES_LENGTH):# SIZE*SIZE):
        if float(std_vector[i]) == 0:
            print i, "is zero"
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
#    eigen_face = U_reduce[:,0].T
#    eigen_face = (eigen_face * std_vector) + mean_vector
#    print eigen_face.shape
#    to_show = eigen_face.reshape(SIZE,SIZE)

#   Display image
#    figure()
#    imshow(to_show)
#    title('PCs # ')
#    gray()
#    show()

    transformed_data = np.dot(data_normalised, U_reduce)
    return [transformed_data, mean_vector, std_vector, U_reduce, k]

def pca_single_image(image_name, mean_vector, std_vector, U_reduce):

    """Function to perform pca on a single image"""
    
    image = cv2.imread(image_name)
    image = preprocess(image, image_name)
    if image is None:
        print "Couldn't load image"
        return [False, 0]
#    print "test image = ", image
    image_array = transform(image)#image.ravel()
    for num in image_array:
        print num,
#    print image_array
#    print "charlie image", image_array
    normalised_image_array = (image_array - mean_vector) / std_vector
    transformed_image_array = np.dot(normalised_image_array, U_reduce)
    print normalised_image_array.shape
    print U_reduce.shape
    print transformed_image_array.shape
    return [transformed_image_array, 1]
    

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
    
    
def compare_images(image, data, k, human_features):

    """Finds the best match"""
    
    global LIMIT
    
    #Finds the least distance between test image and the dataset
    pos = range(LIMIT) # contains the indices of all the images in the dataBASE
    value = [] # will contain the mean square errors later
#    k_comp = k
#    a = np.zeros((10,81))
#    b = np.zeros((10,81))
#    a[:] = np.concatenate((image, np.zeros(81 - k_comp)))#.reshape(9, 9)
#    print image + 1000
#    assert False
    if len(human_features) > 0:
        image = np.concatenate((image, image[human_features]))
#    print "image", image
    for i in range(len(pos)):
#        value.append(scipy.spatial.distance.correlation(image, data[i]))
#        value.append(distance(image, data[i]))
#        value.append(1 - (np.dot(image, data[i]) / math.sqrt(np.dot(image, image) * np.dot(data[i], data[i]))))
        if len(human_features) > 0:
            learnt_data = np.concatenate((data[i], data[i][human_features]))
            value.append(1 - (np.dot(image, learnt_data) / math.sqrt(np.dot(image, image) * np.dot(learnt_data, learnt_data))))
        else:
            value.append(1 - (np.dot(image, data[i]) / math.sqrt(np.dot(image, image) * np.dot(data[i], data[i]))))
#        if i == 1:
#            print "learnt", learnt_data
#            print "dot", 1 - (np.dot(image, learnt_data) / math.sqrt(np.dot(image, image) * np.dot(learnt_data, learnt_data)))
##        b[:] = np.concatenate((data[i], np.zeros(81 - k_comp)))#.reshape(9, 9)
##        sim = ssim(a, b, dynamic_range=a.max() - a.min())
##        if sim > 0:
##            value.append(sim)
##        else:
##            value.append(999)
#        if i == 0:
#            continue
#        k = i
#        while((value[k] < value[k - 1]) and k > 0):# and (value[k] >= 0)):
#            #swap positions
#            pos[k], pos[k - 1] = pos[k - 1], pos[k]
#            #swap values
#            value[k], value[k - 1] = value[k - 1], value[k]
#            k -= 1
    pos, value = izip(*sorted(izip(pos, value), key=lambda x: x[1]))
    print
    #Now pos will contain the indexes to images in ascending order of distance
    return pos, value


def find_similar(image, data, clf, k):

    """Finds the best match"""
    
    global LIMIT

    #Finds the least distance between test image and the dataset
    pos = range(LIMIT) # contains the indices of all the images in the dataBASE
#    value = [] # will contain the mean square errors later
    test_data = np.zeros((LIMIT, k))
    for i in range(LIMIT):
#        test_data[i] = np.concatenate((image, data[i]))
        test_data[i] = abs(np.subtract(image, data[i]))
#        if i == 2:
#            print "saaaar ", test_data[i]
#    for random forests
#    probabilities = clf.predict_proba(test_data)
#    probabilities = probabilities[:,1]    
#    pos, probabilities = izip(*sorted(izip(pos, probabilities),reverse=True, key=lambda x: x[1]))
#    for linearRegression
    values = clf.predict_proba(test_data)
    values = values[:,0]
    pos, values = izip(*sorted(izip(pos, values),reverse=True, key=lambda x: x[1]))
    
#    for i in range(len(pos)):
#        value.append(distance(image, data[i]))
#        if i == 0:
#            continue
#        k = i
#        while((value[k] < value[k - 1]) and k > 0):
#            #swap positions
#            pos[k], pos[k - 1] = pos[k - 1], pos[k]
#            #swap values
#            value[k], value[k - 1] = value[k - 1], value[k]
#            k -= 1
#    print
    #Now pos will contain the indexes to images in ascending order of distance
    return pos, values

#def store_data(data):
#    
#    pickle.dump(data, open(MEMORY_DATA, 'wb'))
    
def render_images(pos, test_img_path):
    
    global RESULTS
    html = """<html>
        <head><title>Similar images</title></head>

        <body>
        <h1 align="center">Test image</h1>
        <p align="center"><img src="{{ test_image }}" alt="test image" height="250" width="250"></p>

        <h1 align="center">Similar images</h1>

        <ol>
        {% for item in list %}
            <li><img src="{{ item }}" alt="similar image" height="250" width="250"></li>
            </br>
        {% endfor %}
        </ol>

        </body>
        </html>
        """    
    t = template.Template(html)
    pos = [FOLDER + "/" + str(i) + ".jpg" for i in pos]
    c = template.Context({'list': pos, 'test_image': test_img_path})
    html = t.render(c)
    f = open(RESULTS + ".html", 'w')
    f.write(html)


def create_training_data(transformed_data, k):

    """Reads a file and creates training vectors"""
    
    print "entered training"
    filename = "train"
    pickle_filename = "train.dat"
    lines = open(filename, "r").readlines()
    count = len(lines)
    print "count = ", count
#    Concatenated vectors
    X = np.zeros((count, k)).astype('float32')
    y = np.zeros(count)
    i = 0
    for line in lines:
        line = line.strip()
        line = line.split()
        a = transformed_data[int(line[0]) - 1]#.ravel()
        b = transformed_data[int(line[1]) - 1]#.ravel()
        if np.sum(a) == 0:
            print "saaaar", line[0]
        if np.sum(b) == 0:
            print "saaaaar", line[1]
        X[i] = abs(np.subtract(transformed_data[int(line[0]) - 1], transformed_data[int(line[1]) - 1]))
#        X[i] = np.concatenate((transformed_data[int(line[0]) - 1], transformed_data[int(line[1]) - 1]))
        y[i] = int(line[2])
#        if int(line[2]) == 1:
#            y[i] = 0
#        else:
#            y[i] = 1 
        i += 1
    training_data = [X, y]
#    print y
    pickle.dump(training_data, open(pickle_filename, 'wb'))
    
def main():

    global NO_SIMILAR, RESULTS
    test_img_path = sys.argv[1]
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
    try:
        [transformed_data, mean_vector, std_vector, U_reduce, k] = pickle.load(open(MEMORY_DATA, 'rb'))
        print "data found"
    except:    
        print "no data found"
        [data, images] = init()
        [transformed_data, mean_vector, std_vector, U_reduce, k] = run_pca(data)
        data = [transformed_data, mean_vector, std_vector, U_reduce, k]
        pickle.dump(data, open(MEMORY_DATA, 'wb'))

    print "k is ", k
#    print transformed_data[0]
#    create_training_data(transformed_data, k)
#    [X, y] = pickle.load(open("train.dat", 'rb'))
    
#    RandomForestClassifier    
#    clf = RandomForestClassifier(n_estimators=NO_TREES)#svm.SVC()
#    clf = LinearRegression()
#    clf = svm.SVC(probability=True)
#    clf.fit(X,y)
    

#    print clf.feature_importances_    

#    image_name = FOLDER + "/304"

    [transformed_image, flag] = pca_single_image(test_img_path, mean_vector, std_vector, U_reduce)
#    print "TI\n", transformed_image
#    print "charlie transforemed", transformed_image
    if flag != False:
    
        try:
            human_features = pickle.load(open(MEMORY_REINFORCE, 'rb'))
        except:    
            human_features = np.array([])
        
        print "human_features", human_features
        pos, value = compare_images(transformed_image, transformed_data, k, human_features)
#        pos, value = find_similar(transformed_image, transformed_data, clf, k)        
        pos = np.array(pos)
        pos += 1
        print "count", count_useless_images
        print "pos\n", pos[0:100]
        print "value \n", value[0:100]
        pos = pos[0:NO_SIMILAR]
        
    #    pos += 1
    #    print pos
        test_img_path = test_img_path
    #    To create a html page
        render_images(pos, test_img_path)
        
        no_times = 1
        while no_times >= 0:
            
            ranks = raw_input("Enter the ranks of image you feel is similar, space").split()
            for rank in ranks:
                index = pos[int(rank) - 1] - 1
                diff = np.array(abs(transformed_data[index] - transformed_image))
                cos_sim = 1 - (np.dot(transformed_image, transformed_data[index]) / math.sqrt(np.dot(transformed_image, transformed_image) * np.dot(transformed_data[index], transformed_data[index])))
                main_indices = np.array(range(k))
                main_indices, diff = izip(*sorted(izip(main_indices, diff), key=lambda x: x[1]))
                human_features = np.union1d(human_features, np.array(main_indices[0:int(k/FACTOR_REINFORCEMENT_FEATURES)])).astype('int')
                
            print "human_features", human_features
            pos, value = compare_images(transformed_image, transformed_data, k, human_features)
            pos = np.array(pos)
            pos += 1
            
            print pos[0:50]
            print value[0:50]
            pos = pos[0:10]
            render_images(pos, test_img_path)
            pickle.dump(human_features, open(MEMORY_REINFORCE, 'wb'))
            no_times -= 1
        
    #    for i in range(NO_SIMILAR):
    #        figure()
    #        my_image = imread(FOLDER + "/" + str(int(pos[i])) + ".jpg") 
    #        imshow(flipud(my_image))
    #        title(str(i + 1))
    #    show()
    else:
        f = open(RESULTS + ".html", "w")
        f.write("<h4>Sorry, couldn't load the input image. Try with a different one.</h3>'")
    
    
#    index = 90
#    diff = np.array(abs(transformed_data[index] - transformed_image))
#    cos_sim = 1 - (np.dot(transformed_image, transformed_data[index]) / math.sqrt(np.dot(transformed_image, transformed_image) * np.dot(transformed_data[index], transformed_data[index])))
##    print np.mean(diff)
#    main_indices = np.array(range(k))
##    for it in range(len(diff)):
##        if diff[it] < np.mean(diff):
##            main_indices.append(it)
##    print len(main_indices)
##    print cos_sim
##    print diff.shape
##    print main_indices.shape
#    main_indices, diff = izip(*sorted(izip(main_indices, diff), key=lambda x: x[1]))
#    human_features = np.array(main_indices[0:int(k/5)])
##    print human_features
#    pos, value = compare_images(transformed_image, transformed_data, k)
#    pos = np.array(pos)
#    pos += 1
#    
#    print pos[0:50]
#    print value[0:50]
#    pos = pos[0:10]
#    render_images(pos, test_img_path)
main()
