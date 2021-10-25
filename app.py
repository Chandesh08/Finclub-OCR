from flask import Flask, render_template, Response, request, session
import cv2
import string
import random
import numpy as np
import base64
from pyzbar import pyzbar
from datetime import datetime
from pytesseract import pytesseract
import re
import sqlalchemy as db
import pandas as pd

pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

engine = db.create_engine('sqlite:///fregistration.sqlite??check_same_thread=False')  # Create test.sqlite automatically
connection = engine.connect()
metadata = db.MetaData()

emp = db.Table('emp', metadata,
               db.Column('Id', db.Integer(), primary_key=True, autoincrement=True),
               db.Column('fname', db.String(255), nullable=False),
               db.Column('sname', db.String(255), nullable=False),
               db.Column('mname', db.String(255), nullable=True),
               db.Column('gender', db.CHAR(1), nullable=False),
               db.Column('title', db.String(5), nullable=False),
               db.Column('dob', db.String(20), nullable=False),
               db.Column('nic', db.CHAR(14), nullable=False),
               db.Column('niccode', db.String(15), nullable=False),
               db.Column('email', db.String(100), nullable=False),
               db.Column('mob', db.String(25), nullable=False),
               db.Column('role', db.String(15), nullable=False),
               db.Column('ad_name', db.String(35), nullable=True),
               db.Column('address', db.String(100), nullable=True),
               )

metadata.create_all(engine)  # Creates the table
connection.close()


def ocr(i, j):
    per = 25
    myData = []

    def get_text(img_name, type):
        image = img_name
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([130, 175, 90])
        mask = cv2.inRange(hsv, lower, upper)
        # Invert image and OCR
        invert = 255 - mask
        if type == 'NIC Code':
            data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
            myData.append(re.sub('[^0-9]+', '', data))
        else:
            data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
            myData.append(re.sub('[^A-Za-z- ]+', '', data))
        return data

    roi = [[(210, 70), (600, 100), 'text', 'Surname'],
           [(210, 115), (600, 145), 'text', 'Firstname'],
           [(210, 165), (600, 195), 'text', 'Surname at Birth'],
           [(210, 213), (250, 240), 'text', 'Gender']]

    imgQ = cv2.imread('template.png')
    imgQ = cv2.resize(imgQ, (1920, 1080))
    h, w, c = imgQ.shape
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)

    img = cv2.imread(i)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:40], None, flags=2)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    imgScan = cv2.resize(imgScan, (w // 3, h // 3))
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (255, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

        if r[2] == 'text':
            # print(f'{r[3]} : {pytesseract.image_to_string(th)}')
            # myData.append(pytesseract.image_to_string(th))
            print(r[3] + ': ' + re.sub('[^A-Za-z0-9- ]+', '', get_text(imgCrop, r[3])))

    # --------------------------------------------------------------------------------------------------------

    per = 5

    roi = [[(400, 235), (620, 280), 'text', 'NIC Code']]

    imgQ = cv2.imread('backtemp1.png')
    imgQ = cv2.resize(imgQ, (1920, 1080))
    h, w, c = imgQ.shape
    imgSc = cv2.resize(imgQ, (w // 3, h // 3))
    # cv2.imshow('y10', imgSc)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)

    img = cv2.imread(j)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:1], None, flags=2)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 7.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    imgScan = cv2.resize(imgScan, (w // 3, h // 3))
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (255, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

        if r[2] == 'text':
            print(r[3] + ': ' + re.sub('[^0-9- ]+', '', get_text(imgCrop, r[3])))

        barcodes = pyzbar.decode(imgShow)

        # loop over the detected barcodes
        for barcode in barcodes:
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            text = "{} ({})".format(barcodeData, barcodeType)
            i = barcodeData[1:]
            i = i[0:6]
            print(i)
            ii = i[-2:]
            iii = i[:4]
            if int(ii) > 25:
                i11 = '19' + ii
            else:
                i11 = '20' + ii
            i = iii + i11
            print(i)
            s_datetime = datetime.strptime(i, '%d%m%Y')

            print("Date Of Birth: " + s_datetime.strftime("%d %b %Y"))
            myData.append(s_datetime.strftime("%d %b %Y"))
            print("NIC No.: " + barcodeData)
            myData.append(barcodeData)
    print(myData)

    return myData


def tele(q):
    per = 25

    roi = [[(10, 280), (400, 450), 'text', 'info']]

    imgQ = cv2.imread('Mytproof.png')
    h, w, c = imgQ.shape
    imgQ = cv2.resize(imgQ, (1080, 1920))
    h, w, c = imgQ.shape
    imgSc = cv2.resize(imgQ, (w // 3, h // 3))
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)

    def get_text(img_name):
        image = img_name
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Remove shadows
        dilated_img = cv2.dilate(gray, np.ones((2, 2), np.uint8))
        # Threshold using Otsu's
        th = cv2.threshold(dilated_img, 0, 255, cv2.THRESH_OTSU)[1]
        data = pytesseract.image_to_string(th, lang='eng', config='--psm 6')
        return data

    img = cv2.imread(q)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:40], None, flags=2)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    q = []

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (255, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        h1, w1, c1 = imgCrop.shape
        imgCrop = cv2.resize(imgCrop, (w1 * 4, h1 * 4), interpolation=cv2.INTER_NEAREST)
        if r[2] == 'text':
            w1 = get_text(imgCrop)
            io = w1.split("\n", 1)[:1]
            import re
            jkl = re.sub('[^A-Za-z ]+', '', str(io))
            print(jkl)
            q.append(jkl)
            w2 = w1.split("\n", 1)[1]
            q.append(w2)
            print(q)
    return q


def ceb(q):
    per = 25

    roi = [[(505, 78), (900, 160), 'text', 'info']]

    imgQ = cv2.imread('CEBproof.png')
    h, w, c = imgQ.shape
    imgSc = cv2.resize(imgQ, (w // 3, h // 3))
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)

    def get_text(img_name):
        image = img_name
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 1), np.uint8)
        # Remove shadows
        img_erosion = cv2.erode(gray, kernel, iterations=1)
        # Threshold using Otsu's
        th = cv2.threshold(img_erosion, 0, 255, cv2.THRESH_OTSU)[1]
        data = pytesseract.image_to_string(th, lang='eng', config='--psm 6')
        return data

    img = cv2.imread(q)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:40], None, flags=2)
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    q = []

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (255, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        h1, w1, c1 = imgCrop.shape
        imgCrop = cv2.resize(imgCrop, (w1 * 2, h1 * 2), interpolation=cv2.INTER_NEAREST)
        if r[2] == 'text':
            w1 = get_text(imgCrop)
            io = w1.split("\n", 1)[:1]
            import re
            jkl = re.sub('[^A-Za-z ]+', '', str(io))
            print(jkl)
            q.append(jkl)
            w2 = w1.split("\n", 1)[1]
            q.append(w2)
            print(q)
    return q


def cwa(q):
    per = 25
    roi = [[(5, 120), (350, 160), 'text', 'name'],[(5, 150), (450, 200), 'text', 'address']]
    imgQ = cv2.imread('CWAproof.png')
    h, w, c = imgQ.shape
    imgSc = cv2.resize(imgQ, (w // 3, h // 3))
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)

    def get_text(img_name, qwerty):
        image = img_name
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 1), np.uint8)
        # Remove shadows
        img_erosion = cv2.erode(gray, kernel, iterations=1)
        # Threshold using Otsu's
        th = cv2.threshold(img_erosion, 0, 255, cv2.THRESH_OTSU)[1]
        cv2.imshow(qwerty, th)
        data = pytesseract.image_to_string(th, lang='eng', config='--psm 6')
        return data

    img = cv2.imread(q)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:40], None, flags=2)
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    q = []

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (255, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        h1, w1, c1 = imgCrop.shape
        imgCrop = cv2.resize(imgCrop, (w1 * 2, h1 * 2), interpolation=cv2.INTER_NEAREST)
        if r[2] == 'text':
            w1 = get_text(imgCrop, r[3])
            q.append(w1)
            print(q)
    cv2.waitKey(0)
    return q


def id_generator(size=14, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


app = Flask(__name__)
app.secret_key = 'ocr'

directory = []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/choose')
def choose():
    return render_template('choose.html')


@app.route('/bills')
def bills():
    return render_template('utilitybills.html')


@app.route('/error')
def error():
    return render_template('400.html')


@app.route('/submit', methods=['POST'])
def submit():
    fname = request.form['fname']
    sname = request.form['sname']
    mname = request.form['mname']
    gender = request.form['gender']
    title = request.form['title']
    dob = request.form['dob']
    nic = request.form['nic']
    niccode = request.form['niccode']
    email = request.form['email']
    mob = request.form['mob']
    role = request.form['role']
    query = db.insert(emp).values(fname=fname, sname=sname, mname=mname, gender=gender, title=title, dob=dob, nic=nic,
                                  niccode=niccode, email=email, mob=mob, role=role, ad_name='', address='')
    connection = engine.connect()
    ResultProxy = connection.execute(query)
    connection.close()
    session['fname'] = fname
    session['sname'] = sname
    session.pop('di', None)
    session.pop('di1', None)
    return render_template('submit.html')

@app.route('/submit1', methods=['POST'])
def submit1():
    name = request.form['name']
    address = request.form['address']
    query = db.update(emp).where(emp.c.fname==session['fname'], emp.c.sname==session['sname']).values(ad_name=name, address=address)
    connection = engine.connect()
    ResultProxy = connection.execute(query)
    connection.close()
    return render_template('submit1.html')

@app.route('/takeimage', methods=['POST'])
def takeimage():
    name1 = request.form['front']
    img = data_uri_to_cv2_img(name1)
    session['di'] = id_generator()
    directory.append(session['di'])
    cv2.imwrite('static/Front/FrontNIC' + session['di'] + '.jpg', img)
    return Response(status=200)


@app.route('/bnic', methods=['POST'])
def bnic():
    name = request.form['back']
    img = data_uri_to_cv2_img(name)
    session['di1'] = id_generator()
    directory.append(session['di1'])
    cv2.imwrite('static/Back/BackNIC' + session['di1'] + '.jpg', img)
    return Response(status=200)


@app.route('/telecom', methods=['POST'])
def telecom():
    name = request.form['tel']
    img = data_uri_to_cv2_img(name)
    session['di3'] = id_generator()
    cv2.imwrite('static/Telecom/Tel' + session['di3'] + '.jpg', img)
    return Response(status=200)


@app.route('/electricity', methods=['POST'])
def electricity():
    name = request.form['elec']
    img = data_uri_to_cv2_img(name)
    session['di4'] = id_generator()
    cv2.imwrite('static/CEB/CEB' + session['di4'] + '.jpg', img)
    return Response(status=200)


@app.route('/water', methods=['POST'])
def water():
    name = request.form['water']
    img = data_uri_to_cv2_img(name)
    session['di5'] = id_generator()
    cv2.imwrite('static/CWA/CWA' + session['di5'] + '.jpg', img)
    return Response(status=200)


@app.route('/address')
def address():
    if not session.get("di3") is None:
        img33 = 'static/Telecom/Tel' + session['di3'] + '.jpg'
        y = tele(img33)
    elif not session.get("di4") is None:
        img33 = 'static/CEB/CEB' + session['di4'] + '.jpg'
        y = ceb(img33)
    elif not session.get("di5") is None:
        img33 = 'static/CWA/CWA' + session['di5'] + '.jpg'
        y = cwa(img33)
    session.pop('di3', None)
    session.pop('di4', None)
    session.pop('di5', None)
    return render_template('checkaddress.html', data=y)


@app.route('/test123')
def test123():
    connection = engine.connect()
    results = connection.execute(db.select([emp])).fetchall()
    df = pd.DataFrame(results)
    df.columns = results[0].keys()
    connection.close()
    dictionaryObject = df.to_dict()
    return render_template('data.html', tables=[df.to_html(classes='data')], titles=df.columns.values)


@app.route('/photo')
def photo():
    print(session['di'] + session['di1'])
    img11 = 'static/Front/FrontNIC' + session['di'] + '.jpg'
    img22 = 'static/Back/BackNIC' + session['di1'] + '.jpg'
    y = ocr(img11, img22)
    return render_template('check.html', data=y)


@app.errorhandler(500)
def page_not_found(e):
    return render_template('400.html'), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
