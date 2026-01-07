import cv2 as cv
from cv2 import data
import json

# Get image filename from user
image_filename = input("Enter your image filename (with extension): ")

# Load the image
image = cv.imread(f'images/{image_filename}')

# Create haarcascade model for classifiers face detection image
face_cascade = cv.CascadeClassifier(data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Detect the faces in image
faces = face_cascade.detectMultiScale(gray_image, 2.3, 5)

# get data coordinates of detected faces
DATA_COORDINATES = {"image": f"{image_filename}", "faces": []}

# Draw a rectangles around faces
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Store coordinates in the data structure
    DATA_COORDINATES["faces"].append({
        "coordinate": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
        }
    })
    
    # Save coordinates data to JSON file
    with open("faces_coordinates.json", "w") as json_file:
        json.dump(DATA_COORDINATES, json_file, indent=3)

# Display the image with detected faces
final_image = cv.imwrite('images/result_detected.jpg', image)
print("Image saved successfully with detected faces.") if final_image else print("Failed for saving image result detected.")
cv.waitKey(0)
cv.destroyAllWindows()