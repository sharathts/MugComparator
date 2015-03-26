import cv2
import numpy as np
import random

FOLDER = "data"
LIMIT = 10 #limits the number of images to be considered as part of the database
SIZE = 200

def mse(x, y):
    
    """Find the mean absolute error between two column vectors x and y"""
    
    return sum([abs(x[i] - y[i]) for i in range(len(y))])

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
            
data = np.zeros(shape=(LIMIT, SIZE*SIZE))
images = []

for imname in range(1, LIMIT + 1):

    img = cv2.imread(FOLDER + "/" + str(imname) + ".jpg")
    img = preprocess(img)
    images.append(img)
    data[imname - 1] = normalize_and_flatten(img)

print data


#select random image to be test image
test_img = cv2.imread(FOLDER + "/" + str(random.randint(1, LIMIT + 1)) + ".jpg")
test_img = preprocess(test_img)
cv2.imshow("test", test_img)
test_img = normalize_and_flatten(test_img)

#Finds the least mse between test image and the dataset
pos = range(LIMIT)
value = []
for i in range(len(pos)):
    value.append(mse(test_img, data[i]))
    print i
    if i == 0:
        continue
    k = i
    while((value[k] < value[k - 1]) and k >= 0):
        #swap positions
        pos[k], pos[k - 1] = pos[k - 1], pos[k]
        #swap values
        value[k], value[k - 1] = value[k - 1], value[k]
        k -= 1

#Now pos will contain the indexes to images in ascending order of mse        
print pos  
for i in range(10):
    cv2.imshow(str(i), images[pos[i]])
cv2.waitKey(0)
cv2.destroyAllWindows()
