import cv2
import numpy as np
import random

FOLDER = "data"
LIMIT = 10 #limits the number of images to be considered as part of the database
SIZE = 150
TOTAL_NO_IMAGES = 300
NO_SIMILAR = 5

NAME = random.sample(range(1, TOTAL_NO_IMAGES), LIMIT)
NAME = [str(i) for i in NAME]

def mse(x, y, z):
    
    """Find the mean absolute error between two column vectors x and y and add z to this"""
    
    return sum([(abs(x[i] - y[i]) + z[i]) for i in range(len(y))])

def preprocess(image):

    """Converts the image to gray scale and also resizes the image"""
    
    global SIZE
    image = cv2.resize(image, (SIZE, SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def normalize_and_flatten(image):
    
    """Some normalization is applied on the image and then the image is converted into a row vector so that it can be dealt with easily"""
    
    image = image / 255.0 #For now, we apply a very naive method of normalization
    image = image.ravel()
    return image

def recognize_image(test_img, data, reinforce_data):
    
    """Given the test image return the index of 10 most similar hits"""
    
    global LIMIT
    #Finds the least mse between test image and the dataset
    pos = range(LIMIT) # contains the indices of all the images in the database
    value = [] # will contain the mean square errors later
    for i in range(len(pos)):
        value.append(mse(test_img, data[i], reinforce_data[i]))
        print i,
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
    return pos, value
    
def init():

    data = np.zeros(shape=(LIMIT, SIZE*SIZE))
    images = []
    for itr in range(LIMIT):

        img = cv2.imread(FOLDER + "/" + NAME[itr] + ".jpg")
        img = preprocess(img)
        images.append(img)
        data[itr - 1] = normalize_and_flatten(img)
    
    return [data, images]

def display_results(images, pos):
    
    #Now pos will contain the indexes to images in ascending order of mse        
    for i in range(10):
        cv2.imshow(str(i + 1), images[pos[i]])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def learn(test_img, data, reinforce_data, value, mismatch_list):
    
    per_pix_error = value / (SIZE * SIZE)
    print "learning now", per_pix_error
    for i in mismatch_list:
        error = [abs(test_img[j] - data[i][j]) for j in range(len(data[i]))]
        error= [per_pix_error / (er + 0.1) for er in error]
        k = 0
        for k in range(len(reinforce_data[i])):
            reinforce_data[i][k] += error[k]
    return reinforce_data
    
def main():
    
    data, images = init()
    
    #select random image to be test image
    test_img = cv2.imread(FOLDER + "/" + str(random.randint(1, TOTAL_NO_IMAGES)) + ".jpg")
    test_img = preprocess(test_img)
    cv2.imshow("test", test_img)
    test_img = normalize_and_flatten(test_img)
    reinforce_data = np.zeros(shape=(LIMIT, SIZE*SIZE))
    i = 2
    while i >= 0:
        pos, value = recognize_image(test_img, data, reinforce_data)
        pos = pos[0:NO_SIMILAR]
        print pos
        print value
        value.sort()
        print value
        reinforce_data = learn(test_img, data, reinforce_data, max(value), [pos[1]])
        print "Next iter"
#        print reinforce_data
        i -= 1
main()
