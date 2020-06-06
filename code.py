import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import time



print("Please bring your face about 15cm to the Web CAM")
print("Wait for the Web Cam to activate")



# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
def photo():
    videoCaptureObject = cv2.VideoCapture(0)
    i=0
    if i<1:
        ret,frame = videoCaptureObject.read()
        cv2.imwrite("test_photo.jpg",frame)
        i=i+1
        
    videoCaptureObject.release()
    cv2.destroyAllWindows()
photo()
image = Image.open('test_photo.jpg')

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
  

#turn the image into a numpy array
image_array = np.asarray(image)

# display the resized image
#image.show()


# pip install pillow


#Loading
import progressbar
from time import sleep
bar = progressbar.ProgressBar(maxval=20, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i in range(10):
    bar.update(i+1)
    sleep(0.1)
bar.finish()


# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
prediction=prediction.flatten()


if prediction[0]>prediction[1]:
    print("Masked")
else:
    print("Non Masked")

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(image) 
plt.show()

