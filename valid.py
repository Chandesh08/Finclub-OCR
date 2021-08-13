import cv2
import numpy as np
import pandas as pd
from pytesseract import pytesseract
from imutils.perspective import four_point_transform
from pyzbar import pyzbar
from datetime import datetime

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

startTime = datetime.now()

image = cv2.imread('img/test.jpg')
scale_percentage = 40
width = int(image.shape[1] * scale_percentage/100)
height = int(image.shape[0] * scale_percentage/100)
dsize = (width, height)
image = cv2.resize(image, dsize, interpolation=cv2.INTER_NEAREST)
original_image = image.copy()

# convert the image to grayscale, blur it, and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
#cv2.imshow('einfn', edged)

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screen_cnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)

    if len(approx) == 4:
        screen_cnt = approx
        break

# Draw ROI
#cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 3)
scale_percentage = 40
width = int(image.shape[1] * scale_percentage/100)
height = int(image.shape[0] * scale_percentage/100)
dsize = (width, height)
rr = cv2.resize(original_image, dsize, interpolation=cv2.INTER_NEAREST)
cv2.imshow("image", rr)
#cv2.imshow("ROI", image)



#gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
gray = four_point_transform(image, screen_cnt.reshape(4, 2))
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
dimensions = gray.shape
print(dimensions)

gray = cv2.resize(gray, (989,609))
dimensions = gray.shape
print(dimensions)

#cv2.imshow("transformedaaa", gray)
#cv2.waitKey(0)

# Remove shadows
dilated_img = cv2.dilate(gray, np.ones((5, 5), np.uint8))
bg_img = cv2.medianBlur(dilated_img, 21)
diff_img = 255 - cv2.absdiff(gray, bg_img)
norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Threshold using Otsu's
th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_OTSU)[1]

#Cropping
crop_image = th[120:175, 300:870]
#cv2.imshow("cropped", crop_image)

crop_image2 = th[200:250, 300:870]
#cv2.imshow("cropped1", crop_image2)

crop_image3 = th[290:350, 300:870]
#cv2.imshow("cropped2", crop_image3)

crop_image4 = th[370:420, 320:370]
#cv2.imshow("cropped3", crop_image4)

crop_image5 = th[375:420, 440:670]
#cv2.imshow("cropped4", crop_image5)

crop_image1 = th[552:610, 12:370]
#cv2.imshow("cropped_", crop_image1)
#cv2.waitKey(0)

print("Execution Time: {}".format(datetime.now() - startTime))
data = []
text = pytesseract.image_to_string(crop_image1)
test= str(text[:-1])
data.append(text[:-1])
print('NIC: '+text[:-1])
text = pytesseract.image_to_string(crop_image)  
data.append(text[:-1])
print('Surname: '+text[:-1])
text = pytesseract.image_to_string(crop_image2)  
data.append(text[:-1])
print('First Name: '+text[:-1])
text = pytesseract.image_to_string(crop_image3)  
data.append(text[:-1])
print('Surname at Birth: '+text[:-1])
text = pytesseract.image_to_string(crop_image5)  
data.append(text[:-1])
print('Date of Birth: '+text[:-1])

text = pytesseract.image_to_string(crop_image4, config=("-c tessedit"
                  "_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                  " --psm 10"
                  " -l osd"
                  " "))

from difflib import SequenceMatcher
s = SequenceMatcher(None, "M", "M")
s.ratio()
print(s)
  
data.append(text[:-1])
print('Gender: '+text[:-1])

df1 = pd.DataFrame([data], columns=['NIC','Surname', 'Firstname', 'Surname at birth', 'Date of birth', 'Gender'])
df1.to_excel("Book1.xlsx") 

#---------------------------------------------------------------------------------------------------------------------------

image = cv2.imread('img/barcode2.jpg')
scale_percentage = 40
width = int(image.shape[1] * scale_percentage/100)
height = int(image.shape[0] * scale_percentage/100)
dsize = (width, height)
image = cv2.resize(image, dsize, interpolation=cv2.INTER_NEAREST)
original_image = image.copy()

# convert the image to grayscale, blur it, and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screen_cnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)

    if len(approx) == 4:
        screen_cnt = approx
        break

# Draw ROI
#cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 3)
scale_percentage = 40
width = int(image.shape[1] * scale_percentage/100)
height = int(image.shape[0] * scale_percentage/100)
dsize = (width, height)
rr = cv2.resize(original_image, dsize, interpolation=cv2.INTER_NEAREST)
cv2.imshow("image1", rr)
#cv2.imshow("ROI", image)

#gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
gray = four_point_transform(image, screen_cnt.reshape(4, 2))
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
dimensions = gray.shape
print(dimensions)

gray = cv2.resize(gray, (989,609))
dimensions = gray.shape
print(dimensions)

#cv2.imshow("transformedaaa", gray)

barcodes = pyzbar.decode(image)

# loop over the detected barcodes
for barcode in barcodes:
	# extract the bounding box location of the barcode and draw the
	# bounding box surrounding the barcode on the image
	(x, y, w, h) = barcode.rect
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# the barcode data is a bytes object so if we want to draw it on
	# our output image we need to convert it to a string first
	barcodeData = barcode.data.decode("utf-8")
	barcodeType = barcode.type
	# draw the barcode data and barcode type on the image
	text = "{} ({})".format(barcodeData, barcodeType)
	cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (0, 0, 255), 2)
	# print the barcode type and data to the terminal
	print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))

barcodeData = str(barcodeData)
if barcodeData == test:
#b=data.index(barcodeData)
    print('true')
    print(barcodeData + ' ' + test)
else:
    print('false')
    print(barcodeData + ' ' + test)
 
cv2.waitKey(0)