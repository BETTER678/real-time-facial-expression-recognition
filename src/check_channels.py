import cv2
import os
import numpy as np

train_path = "dataset/train"

emotion = "Happy"

emotion_path = os.path.join(train_path, emotion)

image_name = os.listdir(emotion_path)[0]

image_path = os.path.join(
    emotion_path,
    image_name
)

image = cv2.imread(image_path)

print("Image shape:", image.shape)

# Separate BGR channels
blue = image[:, :, 0]
green = image[:, :, 1]
red = image[:, :, 2]

# Check whether all channels are identical
if np.array_equal(blue, green) and np.array_equal(green, red):
    print("This is a grayscale image stored as 3 channels.")

else:
    print("This is a real color image.")