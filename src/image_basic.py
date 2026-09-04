import cv2

# Read image
image = cv2.imread("images/test.jpg")

# Print original shape
print("Original shape:", image.shape)

# Convert to grayscale
gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# Resize image
resized_image = cv2.resize(
    gray_image,
    (48, 48)
)

# Print resized shape
print("Final shape:", resized_image.shape)

# Display images
cv2.imshow("Original Image", image)

cv2.imshow("Grayscale Image", gray_image)

cv2.imshow("48x48 Image", resized_image)

cv2.waitKey(0)

cv2.destroyAllWindows()