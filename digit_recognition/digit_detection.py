import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import digit_recognition.digit_data_training as digit_data_training
import tensorflow as tf

m = digit_data_training.training_data(tf.keras.datasets.mnist)
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = m.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_number += 1
        