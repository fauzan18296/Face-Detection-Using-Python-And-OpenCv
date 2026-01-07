import cv2 as cv

# Load the image
image = cv.imread('images/image.jpg')

# Create cascade classifiers for face recognition
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Detect the faces in image
faces = face_cascade.detectMultiScale(gray_image, 2.3, 5)

# Draw a rectangles around faces
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected faces
if cv.imwrite('images/result_detected.jpg', image):
    print("Image saved successfully with detected faces.")
cv.waitKey(0)
cv.destroyAllWindows()