import os
import cv2
import matplotlib.pyplot as plt

train_path = "dataset/train"

emotions = os.listdir(train_path)

print("Emotion Classes:")
print(emotions)

print("\nNumber of Training Images:")

for emotion in emotions:

    emotion_path = os.path.join(
        train_path,
        emotion
    )

    number_of_images = len(
        os.listdir(emotion_path)
    )

    print(emotion, ":", number_of_images)


# Display one image from each emotion

plt.figure(figsize=(15, 5))

for index, emotion in enumerate(emotions):

    emotion_path = os.path.join(
        train_path,
        emotion
    )

    image_name = os.listdir(emotion_path)[0]

    image_path = os.path.join(
        emotion_path,
        image_name
    )

    image = cv2.imread(image_path)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    plt.subplot(
        1,
        len(emotions),
        index + 1
    )

    plt.imshow(image_rgb)

    plt.title(emotion)

    plt.axis("off")


plt.show()