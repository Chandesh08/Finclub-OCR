import cv2
import numpy as np
from pytesseract import pytesseract
from datetime import datetime

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

startTime = datetime.now()

def order_corner_points(corners):
    # Separate corners into individual points
    # Index 0 - top-right
    #       1 - top-left
    #       2 - bottom-left
    #       3 - bottom-right
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
    return (top_l, top_r, bottom_r, bottom_l)

def perspective_transform(image, corners):
    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between 
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between 
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in 
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))

def rotate_image(image, angle):
    # Grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

image = cv2.imread('img/unnamed.jpg')
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
        transformed = perspective_transform(original_image, screen_cnt)
        break

# Draw ROI
cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 3)

# Rotate image
#rotated = rotate_image(transformed, -90)

cv2.imshow("image", original_image)
cv2.imshow("ROI", image)
cv2.imshow("transformed", transformed)
cv2.waitKey(0)


#Resize image
scale_percentage = 200
width = int(transformed.shape[1] * scale_percentage/100)
height = int(transformed.shape[0] * scale_percentage/100)
dsize = (width, height)
output1 = cv2.resize(transformed, dsize, interpolation=cv2.INTER_NEAREST)



norm_imd = cv2.normalize(output1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

dilated_img = cv2.dilate(norm_imd[:,:,1], np.ones((2,2), np.uint8))
bg_img = cv2.medianBlur(dilated_img,1)
gray = cv2.cvtColor(norm_imd, cv2.COLOR_BGR2GRAY)

th = cv2.threshold(gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_BINARY, 10, 9)[1]
cv2.imshow('th', th)


# predict tesseract
lang = "eng+nld"
config = "--psm 11 --oem 3"
out_rgb = cv2.cvtColor(th, cv2.COLOR_BGR2RGB)

# uncomment to see raw prediction
# print(pytesseract.image_to_string(out_rgb, lang=lang, config=config))


img_data = pytesseract.image_to_data(
    out_rgb,
    lang=lang,
    config=config,
    output_type=pytesseract.Output.DATAFRAME,
)
img_conf_text = img_data[["conf", "text"]]
img_valid = img_conf_text[img_conf_text["text"].notnull()]
img_words = img_valid[img_valid["text"].str.len() > 1]

# to see confidence of one word
# word = "Gulfaraz"
# print(img_valid[img_valid["text"] == word])

all_predictions = img_words["text"].to_list()
print(all_predictions)

confidence_level = 90

img_conf = img_words[img_words["conf"] > confidence_level]
predictions = img_conf["text"].to_list()

# uncomment to see confident predictions
# print(predictions)
print("Execution Time: {}".format(datetime.now() - startTime))









cv2.imshow("test", bg_img)
cv2.waitKey(0)


text = pytesseract.image_to_string(bg_img)  
print(text[:-1])



#
# Clockwise anticlockwise
#