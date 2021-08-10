import numpy as np
from pytesseract import pytesseract
import cv2

# Picture rotation
def rotate_bound(image, angle):
    # Get width and height
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # Extract the rotation matrix sin cos 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # Calculate the new border size of the image
    nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
    nH = h
 
    # Adjust the rotation matrix
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC)
 
 # Get image rotation angle
def get_minAreaRect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    return cv2.minAreaRect(coords)
 
image_path = "img/unnamed.jpg"
image = cv2.imread(image_path)
angle = get_minAreaRect(image)[-1]
rotated = rotate_bound(image, angle)

scale_percentage = 200

width = int(rotated.shape[1] * scale_percentage/100)
height = int(rotated.shape[0] * scale_percentage/100)

dsize = (width, height)

dilated_img = cv2.dilate(rotated[:,:,1], np.ones((2,2), np.uint8))
bg_img = cv2.medianBlur(dilated_img,21)

diff_img = 255 - cv2.absdiff(rotated[:,:,1], bg_img)
output1 = cv2.resize(diff_img, dsize, interpolation=cv2.INTER_NEAREST)
norm_imd = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
cv2.imshow('norm_img', cv2.resize(norm_imd, (0,0), fx = 0.5, fy=0.5))

th = cv2.threshold(norm_imd, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('th', th)

img_con, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(th)
cv2.drawContours(mask, img_con, -1, (0, 255, 0))
out = np.zeros_like(th)
out[mask == 0] = th[mask == 0]
cv2.imshow('con', out)

output = cv2.resize(norm_imd, dsize, interpolation=cv2.INTER_NEAREST)
cv2.imshow('Resize',output)

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.tesseract_cmd = path_to_tesseract
text = pytesseract.image_to_string(output)  
print(text[:-1])

# show the output image
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("imput", image)
cv2.imshow("output", rotated)
cv2.waitKey(0)