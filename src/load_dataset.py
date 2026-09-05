import tensorflow as tf


# Dataset paths
train_path = "dataset/train"
val_path = "dataset/val"
test_path = "dataset/test"


# Settings
image_size = (48, 48)
batch_size = 32


# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(

    train_path,

    image_size=image_size,

    batch_size=batch_size,

    color_mode="grayscale"
)


# Load validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(

    val_path,

    image_size=image_size,

    batch_size=batch_size,

    color_mode="grayscale",

    shuffle=False
)


# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(

    test_path,

    image_size=image_size,

    batch_size=batch_size,

    color_mode="grayscale",

    shuffle=False
)


# Print class names
print("\nClass Names:")

print(train_dataset.class_names)


# Check one batch
for images, labels in train_dataset.take(1):

    print("\nImage Batch Shape:")

    print(images.shape)


    print("\nLabel Batch Shape:")

    print(labels.shape)


    print("\nFirst 10 Labels:")

    print(labels[:10].numpy())