import cv2
import numpy as np
import random

FOLDER = "data"
LIMIT = 10 #limits the number of images to be considered as part of the database
SIZE = 100

TOTAL_NO_IMAGES = 300
NO_SIMILAR = 5

NAME = random.sample(range(1, TOTAL_NO_IMAGES), LIMIT)
NAME = [str(i) for i in NAME]
casc_path = 'lbpcascades/lbpcascade_frontalface.xml'#'haarcascades/haarcascade_frontalface_alt.xml'#
face_cascade = cv2.CascadeClassifier(casc_path)

def distance(x, y, z):
    
    """Find the mean absolute error between two column vectors x and y and add z to this. Here z contains the values learnt from the user"""
    
    return sum(np.add(abs(np.subtract(x, y)), z))

def preprocess(image):

    """Converts the image to gray scale and also resizes the image"""
    
    global SIZE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image)
    
    for (x,y,w,h) in faces:
        image = image[y:y+h, x:x+w]
    
    image = cv2.resize(image, (SIZE, SIZE))
    return image

def normalize_and_flatten(image):
    
    """Some normalization is applied on the image and then the image is converted into a row vector so that it can be dealt with easily"""
    
    image = image / 255.0 #For now, we apply a very naive method of normalization
    image = image.ravel()
    image -= image.mean()
    return image

def recognize_image(test_img, data, reinforce_data):
    
    """Given the test image return the index of 10 most similar hits"""
    
    global LIMIT
    #Finds the least distance between test image and the dataset
    pos = range(LIMIT) # contains the indices of all the images in the database
    value = [] # will contain the mean square errors later
    
    for i in range(len(pos)):
        value.append(distance(test_img, data[i], reinforce_data[i]))
        print i
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
    
def init():

    """Load the images and the dataset randomly so as to not hamper training"""    
    
    data = np.zeros(shape=(LIMIT, SIZE*SIZE))
    images = []
    for itr in range(LIMIT):

        img = cv2.imread(FOLDER + "/" + NAME[itr] + ".jpg")
        img = preprocess(img)
        images.append(img)
        data[itr - 1] = normalize_and_flatten(img)
    
    return [data, images]

def display_results(images, pos):
    
    """Displays the most similar images"""
    
            
    for i in range(NO_SIMILAR):
        cv2.imshow(str(i + 1), images[pos[i]])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def learn(test_img, data, reinforce_data, value, mismatch_list):
    
    """Each pixel is penalized accordingly based on the error it made, this penalty is the core of the training process"""
    
    per_pix_error = value / (SIZE * SIZE)
#    print "learning now", per_pix_error
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
        display_results(images, pos)
        s = raw_input("Enter numbers of the images which are not similar(space seperated):").split()
        mismatch_list = [pos[int(x) - 1] for x in s]
        if len(mismatch_list) == 0:
            break
        reinforce_data = learn(test_img, data, reinforce_data, max(value), mismatch_list)
        i -= 1
        
main()
