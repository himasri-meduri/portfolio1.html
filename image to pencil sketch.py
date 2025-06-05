import cv2
image_path = r':C:\Users\Meduri Bhagavan\OneDrive\사진\radha krishna 1.png'
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image file '{image_path}'")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray_image)
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray_image, inverted_blur, scale=256.0)
    cv2.imshow("Original Image", image)
    cv2.imshow("Pencil Sketch", sketch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
