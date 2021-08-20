from flask import Flask, render_template, Response, request
import cv2
import pandas as pd
from pytesseract import pytesseract
from imutils.perspective import four_point_transform
from datetime import datetime
import os
import numpy as np

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

startTime = datetime.now()

face_cascade = cv2.CascadeClassifier('test/haarcascade_frontalface_default.xml')

def ocr(i):
    image=cv2.imread(i)
    scale_percentage = 70
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
    #cv2.imshow("image", original_image)
    #cv2.imshow("ROI", image)

    #Resize image
    scale_percentage = 200
    width = int(image.shape[1] * scale_percentage/100)
    height = int(image.shape[0] * scale_percentage/100)
    dsize = (width, height)

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
    #cv2.imshow("profile", profilepic)

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
    text1 = pytesseract.image_to_string(crop_image1)
    data.append(text1[:-1])
    print('NIC: '+text1[:-1])
    text2 = pytesseract.image_to_string(crop_image)  
    data.append(text2[:-1])
    print('Surname: '+text2[:-1])
    text3 = pytesseract.image_to_string(crop_image2)  
    data.append(text3[:-1])
    print('First Name: '+text3[:-1])
    text4 = pytesseract.image_to_string(crop_image3)  
    data.append(text4[:-1])
    print('Surname at Birth: '+text4[:-1])
    text5 = pytesseract.image_to_string(crop_image5)  
    data.append(text5[:-1])
    print('Date of Birth: '+text5[:-1])
    
    
    profilepic = gray[170:500, 10:300]
    directory = r'C:\Users\acer\Desktop\FinClub\static'
    os.chdir(directory)
    pr = str(text2[:-1])
    pr1 = str(text3[:-1])
    filename = pr1.strip() + ' ' + pr.strip() + '.jpg'
    p = cv2.imwrite(filename,profilepic)
    directory = r'C:\Users\acer\Desktop\FinClub'
    os.chdir(directory)
    print(p)

    text6 = pytesseract.image_to_string(crop_image4, config=("-c tessedit"
                    "_char_whitelist=fmFM"
                    " --psm 10"
                    " -l osd"
                    " "))  
    data.append(text6)
    print('Gender: '+text6[:-1])

    df1 = pd.DataFrame([data], columns=['NIC','Surname', 'Firstname', 'Surname at birth', 'Date of birth', 'Gender'])
    df1.to_excel("Book1.xlsx")
    str1 = ''.join(data)
    return '''
            <head>
                <title>OCR</title>
                <link rel="stylesheet" href="static/style.css">
            </head>
            <body>
                <h2>OCR Procressing</h2>
                <span>
                    <img src="'''+ i +'''" width="700" height="400">
                    <img src="static/'''+ filename +'''" width="400" height="400">
                </span>
                <form>
                    <label for="fname">NIC:</label><br>
                    <input type="text" id="fname" name="fname" value="''' + text1 + '''"><br>
                    <label for="lname">Surname:</label><br>
                    <input type="text" id="lname" name="lname" value="''' + text2 + '''"><br><br>
                    <label for="fname">First name:</label><br>
                    <input type="text" id="fname" name="fname" value="''' + text3 + '''"><br>
                    <label for="lname">Surname at birth:</label><br>
                    <input type="text" id="lname" name="lname" value="''' + text4 + '''"><br><br>
                    <label for="lname">Date of birth:</label><br>
                    <input type="text" id="lname" name="lname" value="''' + text5 + '''"><br><br>
                    <label for="lname">Gender:</label><br>
                    <input type="text" id="lname" name="lname" value="''' + text6 + '''"><br><br>
                </form>
            </body>'''










app = Flask(__name__)
video = cv2.VideoCapture(0)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/takeimage', methods = ['POST'])
def takeimage():
    name = request.form['name']
    _, frame1 = video.read()
    cv2.imwrite('test/static/12.jpg', frame1)
    return Response(status = 200)


def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/photo')
def photo():
    img11 = 'test/static/12.jpg'
    print(img11)
    y = ocr(img11)
    return '''
    <img style="border-radius: 500%;" src='static/12.jpg'>
''' + y


if __name__ == '__main__':
    app.run()
    app.debug = False