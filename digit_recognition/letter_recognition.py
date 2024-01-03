import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import letters_data_training
import tensorflow as tf

m = letters_data_training.training_data()
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_number = 1
while os.path.isfile('letters/letter{}.png'.format(image_number)):
    try:
        img = cv2.imread('letters/letter{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        img = tf.keras.utils.normalize(img, axis=1)  # Normalize the input image
        prediction = m.predict(img)
        
        # If you have 27 classes (including unknown characters), you might want to handle the prediction output accordingly
        predicted_class = np.argmax(prediction)
        if predicted_class == 26:
            print("The letter is unknown.")
        else:
            print("The letter is probably a {}".format(chr(ord('A') + predicted_class)))
        
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print("Error:", e)
    finally:
        image_number += 1