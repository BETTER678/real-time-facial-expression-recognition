import os
import cv2
import numpy as np


# Dataset path
train_path = "dataset/train"


# Select one emotion
emotion = "Happy"


# Get emotion folder path
emotion_path = os.path.join(
    train_path,
    emotion
)


# Select first image
image_name = os.listdir(emotion_path)[0]


# Complete image path
image_path = os.path.join(
    emotion_path,
    image_name
)


# Read image
image = cv2.imread(image_path)


# Original image shape
print("Original shape:", image.shape)


# Convert to grayscale
gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


print("Grayscale shape:", gray_image.shape)


# Check pixel values
print("\nBefore normalization:")

print("Minimum pixel value:", gray_image.min())

print("Maximum pixel value:", gray_image.max())


# Normalize pixel values
normalized_image = gray_image / 255.0


print("\nAfter normalization:")

print("Minimum pixel value:", normalized_image.min())

print("Maximum pixel value:", normalized_image.max())


# Add channel dimension
final_image = np.expand_dims(
    normalized_image,
    axis=-1
)


print("\nFinal image shape:", final_image.shape)