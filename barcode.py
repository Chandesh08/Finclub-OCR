from pyzbar import pyzbar
import cv2
from pytesseract import pytesseract
from imutils.perspective import four_point_transform

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

image = cv2.imread('img/6.jpg')
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
cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 3)
cv2.imshow("image", original_image)
cv2.imshow("ROI", image)

#gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
gray = four_point_transform(image, screen_cnt.reshape(4, 2))
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
dimensions = gray.shape
print(dimensions)

gray = cv2.resize(gray, (989,609))
dimensions = gray.shape
print(dimensions)

cv2.imshow("transformedaaa", gray)

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
 
cv2.waitKey(0)