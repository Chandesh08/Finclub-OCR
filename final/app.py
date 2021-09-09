from flask import Flask, render_template, Response, request
import cv2
from pyzbar import pyzbar
import string
import random
import numpy as np
import base64
import re
from pytesseract import pytesseract
from imutils.perspective import four_point_transform
from datetime import datetime
import os
from deskew import determine_skew
import math
from typing import Tuple, Union
import face_recognition

pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
startTime = datetime.now()


def ocr(i, j):

    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)
        # return the resized image
        return resized

    image = cv2.imread(i)
    x, y, z = image.shape
    print(str(x) + '-' + str(y))

    if x > y:
        print("portrait")
        image = image_resize(image, height=1920)
    else:
        print('lanscaspe')
        image = image_resize(image, height=1080)

    original_image = image.copy()

    x, y, z = image.shape
    print(str(x) + '-' + str(y))
    # convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + \
            abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + \
            abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        print("1234gwaba")
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

    angle = determine_skew(gray)
    print(angle)
    rotated = rotate(gray, angle, (0, 0, 0))

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    def dilate(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    edged = dilate(edged)

    #cv2.imshow('einfn', edged)

    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
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
    #cv2.drawContours(image, [screen_cnt], -1, (0, 120, 120), 5)

    x, y, w, h = cv2.boundingRect(screen_cnt)
    aspect_ratio = float(h)/w
    print(aspect_ratio)

    def rotate_bound(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    if aspect_ratio > 1:
        image = rotate_bound(image, -90)

    def is_image_upside_down(img):
        face_locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, face_locations)
        image_is_upside_down = (len(encodings) == 0)
        return image_is_upside_down

    if is_image_upside_down(image):
        print("rotate to 180 degree")
        image = rotate_bound(image, 180)
    else:
        print("image is straight")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    def dilate(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    edged = dilate(edged)

    #cv2.imshow('einfn', edged)

    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screen_cnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        if len(approx) == 4:
            screen_cnt = approx
            break

    gray = four_point_transform(image, screen_cnt.reshape(4, 2))
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    dimensions = gray.shape
    print(dimensions)

    gray = cv2.resize(gray, (1521, 943), interpolation=cv2.INTER_NEAREST)
    dimensions = gray.shape
    print(dimensions)

    #cv2.imshow("transformedaaa", gray)

    # Remove shadows
    dilated_img = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(
        diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Threshold using Otsu's
    th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_OTSU)[1]

    # Cropping
    crop_image = th[190:255, 450:1500]
    #cv2.imshow("surname", crop_image)

    #cv2.imshow("profile", profilepic)

    crop_image2 = th[320:390, 450:1500]
    #cv2.imshow("firstname", crop_image2)

    crop_image3 = th[453:520, 450:1500]
    #cv2.imshow("sab", crop_image3)

    crop_image4 = th[580:650, 500:590]
    #cv2.imshow("gender", crop_image4)

    date = th[580:650, 690:785]
    #cv2.imshow("date", date)

    month = th[575:650, 775:1100]
    #cv2.imshow("month", month)

    crop_image1 = th[855:935, 10:570]
    #cv2.imshow("nic", crop_image1)

    print("Execution Time: {}".format(datetime.now() - startTime))
    data = []
    text1 = pytesseract.image_to_string(crop_image1)
    text1 = re.sub(r'[^a-zA-Z0-9]', '', text1)
    test = str(text1[:-1])
    data.append(text1[:-1])
    print('NIC: '+text1[:-1])
    text2 = pytesseract.image_to_string(crop_image)
    #text2 = text2.split("\n",2)[2]
    data.append(text2[:-1])
    print('Surname: '+text2[:-1])
    text3 = pytesseract.image_to_string(crop_image2)
    #text3 = text3.split("\n",2)[2]
    data.append(text3[:-1])
    print('First Name: '+text3[:-1])
    text4 = pytesseract.image_to_string(crop_image3)
    data.append(text4[:-1])
    print('Surname at Birth: '+text4[:-1])
    text5 = pytesseract.image_to_string(date, config=("-c tessedit"
                                                      "_char_whitelist=0123456789"
                                                      " --psm 7"
                                                      " 2 osd"
                                                      " "))
    text6 = pytesseract.image_to_string(month)
    data.append(str(text5[:-1]) + ' ' + str(text6[:-1]))
    text7 = str(text5[:-1]) + ' ' + str(text6[:-1])
    print('Date of Birth: ' + str(text7))

    text8 = pytesseract.image_to_string(crop_image4, config=("-c tessedit"
                                                             "_char_whitelist=fFmM"
                                                             " --psm 10"
                                                             " -l osd"
                                                             " "))

    profilepic = gray[280:800, 15:450]
    directory = r'C:\Users\acer\Desktop\webrtc-ocr\static'
    os.chdir(directory)
    pr = str(text2[:-1])
    pr1 = str(text3[:-1])
    filename = pr1.strip() + ' ' + pr.strip() + '.jpg'
    p = cv2.imwrite(filename, profilepic)
    directory = r'C:\Users\acer\Desktop\webrtc-ocr'
    os.chdir(directory)
    print(p)

    data.append(text8[:-1])
    print('Gender: '+text8[:-1])

# ----------------------------------------------------------------------------------------------------------------------------------------------

    image = cv2.imread(j)
    x, y, z = image.shape
    if x > y:
        print("portrait")
        image = image_resize(image, height=1920)
    else:
        print('lanscaspe')
        image = image_resize(image, height=1080)

    original_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    
    
    
    def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + \
            abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + \
            abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        print("1234gwaba")
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

    angle = determine_skew(gray)
    print(angle)
    rotated = rotate(gray, angle, (0, 0, 0))

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    def dilate(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    edged = dilate(edged)

    #cv2.imshow('einfn', edged)

    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
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
    #cv2.drawContours(image, [screen_cnt], -1, (0, 120, 120), 5)

    x, y, w, h = cv2.boundingRect(screen_cnt)
    aspect_ratio = float(h)/w
    print(aspect_ratio)

    def rotate_bound(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    if aspect_ratio > 1:
        image = rotate_bound(image, -90)

    def is_image_upside_down(img):
        face_locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, face_locations)
        image_is_upside_down = (len(encodings) == 0)
        return image_is_upside_down

    
    
    

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    def dilate(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    edged = dilate(edged)

    #cv2.imshow('einfn', edged)

    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screen_cnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        if len(approx) == 4:
            screen_cnt = approx
            break
    
    
    
    
    

    #gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    gray = four_point_transform(image, screen_cnt.reshape(4, 2))
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    dimensions = gray.shape
    print(dimensions)

    gray = cv2.resize(gray, (1521, 943), interpolation=cv2.INTER_NEAREST)
    dimensions = gray.shape
    print(dimensions)

    #cv2.imshow("transformedaaa", gray)
    # cv2.waitKey(0)

    # Remove shadows
    dilated_img = cv2.dilate(gray, np.ones((5, 5), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(
        diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Threshold using Otsu's
    th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_OTSU)[1]

    crop_image = th[670:770, 1100:1450]
    #cv2.imshow("cropped", crop_image)
    #cv2.waitKey(0)
    niccode = pytesseract.image_to_string(crop_image, config=("-c tessedit"
                                                              "_char_whitelist=0123456789"
                                                              " --psm 7"
                                                              " 2 osd"
                                                              " "))
    print(niccode[:-1])

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
        return '''
<!doctype html>
<html lang="en">

<head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <link rel="icon" href="../static/icon.png">
    <title>NIC OCR</title>
    <style>
   body {
            background-image: url(../static/background.jpg);
            background-size: cover;
        }
        input[type=text] {
  width: 100%;
  font-weight: 1000;
            background-color: transparent;
  padding: 8px 12px;
  margin: 8px 0;
  box-sizing: border-box;
  border: none;
  border-bottom: 2px solid orangered;
}
label{
    color: blue;
    font-weight: 1000;
}
        .nav-link {
            font-size: 18px;
            color: black;
            font-weight: 3000;
        }

        .login {
            color: orangered;
        }

        .register {
            color: orangered;
        }

        .img {
            width: 50%;
            border-radius: 200%;
        }

        .center {
            text-align: center;
        }
    </style>
</head>

<body>


    <nav class="navbar navbar-expand-md navbar-light">
        <a class="navbar-brand" href="#">
            <img src="../static/finclub.png" alt="logo">
        </a>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#collapsibleNavbar">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="collapsibleNavbar">
            <ul class="navbar-nav ml-auto">
                <li class="nav-item">
                    <a class="nav-link" href="#"><b>Borrow</b></a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#"><b>Lend</b></a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#"><b>Live loan</b></a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="#"><b>How it works</b></a>
                </li>
                <li class="nav-item">
                    <a class="nav-link login" href="#" style="color: orangered;"><b>Login</b></a>
                </li>

                <!-- Dropdown -->
                <li class="nav-item dropdown">
                    <a class="nav-link dropdown-toggle register" style="color: orangered;" href="#" id="navbardrop"
                        data-toggle="dropdown">
                        <b>Register</b>
                    </a>
                    <div class="dropdown-menu">
                        <a class="dropdown-item" href="#"><b>Borrower</b></a>
                        <a class="dropdown-item" href="#"><b>Lender</b></a>
                    </div>
                </li>
            </ul>
        </div>
    </nav>
    <br>


    
     <div class="col-md-12">
     <div class="row">
     <div class="container-fluid d-flex justify-content-center">
      <img src="static/''' + filename + '''" alt="logo" class="img rounded-circle">
    </div>
    </div>
    </div>
    <div class="col-md-12">
      <div class="form-group">
        <label for="usr">NIC:</label>
        <input type="text" class="form-control" id="nic" name="username" value="''' + text1 + '''">
      </div>
      <div class="form-group">
        <label for="usr">NIC Barcode:</label>
        <input type="text" class="form-control" id="nicb" name="username" value="''' + barcodeData + '''">
      </div>
      <div class="form-group">
        <label for="usr">Back NIC Code:</label>
        <input type="text" class="form-control" id="bnicc" name="username" value="''' + niccode + '''">
      </div>
      <div class="form-group">
        <label for="pwd">First Name:</label>
        <input type="text" class="form-control" id="fn" value="''' + text3 + '''">
      </div>
      <div class="form-group">
        <label for="pwd">Surname:</label>
        <input type="text" class="form-control" id="sn" value="''' + text2 + '''">
      </div>
      <div class="form-group">
        <label for="pwd">Surname at birth:</label>
        <input type="text" class="form-control" id="sab" value="''' + text4 + '''">
      </div>
      <div class="form-group">
        <label for="pwd">Date of birth:</label>
        <input type="text" class="form-control" id="dob" value="''' + text7 + '''">
      </div>
      <div class="form-group">
        <label for="pwd">Gender:</label>
        <input type="text" class="form-control" id="gen" value="''' + text8 + '''">
      </div>
    </div>

    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"
        integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous">
    </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous">
    </script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous">
    </script>
    <script>
    var nic = document.getElementById("nic").value;
    var nicb = document.getElementById("nicb").value;
    if(nic == nicb){
        //alert("ID Number does match!");
    }else{
        //alert("ID Number does not match!");
    }
    </script>
</body>

</html>
'''


def id_generator(size=14, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/choose')
def choose():
    return render_template('choose.html')


@app.route('/error')
def error():
    return render_template('400.html')


@app.route('/takeimage', methods=['POST'])
def takeimage():
    name = request.form['front']
    img = data_uri_to_cv2_img(name)
    global di
    di = id_generator()
    cv2.imwrite('static/FrontNIC' + di + '.jpg', img)
    return Response(status=200)


@app.route('/propic', methods=['POST'])
def propic():
    name = request.form['front']
    img = data_uri_to_cv2_img(name)
    global di1
    di1 = id_generator()
    cv2.imwrite('static/profilepic' + di1 + '.jpg', img)
    return Response(status=200)


@app.route('/bnic', methods=['POST'])
def bnic():
    name = request.form['back']
    img = data_uri_to_cv2_img(name)
    global di2
    di2 = id_generator()
    cv2.imwrite('static/BackNIC' + di2 + '.jpg', img)
    return Response(status=200)


@app.route('/photo')
def photo():
    img11 = 'static/FrontNIC' + di + '.jpg'
    img22 = 'static/BackNIC' + di2 + '.jpg'
    #x = barcode(img22)
    y = ocr(img11, img22)
    return y


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
