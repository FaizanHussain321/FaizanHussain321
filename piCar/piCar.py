from imageDetection import  MotionDetector
from imutils.video import VideoStream
from flask import *
import numpy as np
import threading
import argparse
import datetime
import imutils
import time
import cv2
import RPi.GPIO as GPIO
import picamera


#blah

from time import sleep

GPIO.setmode(GPIO.BOARD)

GPIO.setwarnings(False)

# Motor 1

Motor1A = 16
Motor1B = 18
Motor1E = 22

# motor 2

Motor2A = 11
Motor2B = 13
Motor2E = 15



#array of different objects which the dataset can detect

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

outputFrame = None
lock = threading.Lock()


app = Flask(__name__)
title = "PiCar"

#loads pretrained model from storage

net = cv2.dnn.readNetFromCaffe("/var/www/piCar/MobileNetSSD_deploy.prototxt.txt","/var/www/piCar/MobileNetSSD_deploy.caffemodel")


vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route('/')
def index():
    return render_template('index2.html')

def detect_motion(frameCount):
    # gralb global references to the stream and frame
    #lock variables

    global vs, outputFrame, lock

    #intiliase the detector and the total frames

    md =  MotionDetector(accumWeight=0.1)
    total = 0

    #loop over all the frames from the video stream

    while True:
        #read the next frame from stream and then resize it.
        # next convert the frame to grayscale and blur it

        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(7,7), 0)
        status_text = "Nothing Detected"

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)


       # pass the blob data throught the ntwork and fetch the detections and predicitons

        net.setInput(blob)
        detections = net.forward()


        #loop over all the detections

        for i in np.arange(0, detections.shape[2]):

          #grab the probability associated with the prediciton
            confidence = detections[0, 0, i , 2]

          #filter out the weak detections  by checking that the confidence is greater
          # than the  minimum confidence

            if confidence > args["confidence"]:
          # next th eindex of the class label is extracted from
          #  the detections array then  the x, y coordinates of the
          # bounding box  is calculated
               idx = int(detections[0, 0,  i, 1])
               box = detections[0, 0, i , 3:7] * np.array([w, h, w, h])
               (startX, startY, endX, endY) = box.astype("int")

            #next the prediciton is drawn on the frame

               label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
               cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
               y = startY - 15 if startY - 15 > 15 else startY + 15
               cv2.putText(frame, label, (startX, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()

        cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),

        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        if total > frameCount:

            motion = md.detect(gray)

            # check if motion was found and not null

            if motion is not None:
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),( 0, 0, 255), 2)

                status_text = "Motion Detected"
               # cv2.putText(frame, "Room Status: {}".format(status_text), (10,20),
               # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            else:
               status_text = "no motion Detected"
              # cv2.putText(frame, "Room Status: {}".format(status_text), (10,20),
              # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #update the backgorund model and increment the total number of frames

        md.update(gray)
        total +=1

        with lock:
            outputFrame = frame.copy()




def generate():

    global outputFrame, lock

    while True:

        with lock:

            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue

        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage.tobytes()) + b'\r\n')



@app.route("/video_feed")
def video_feed():

    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

def setup():

    GPIO.setmode(GPIO.BOARD)

    Motor1A = 16
    Motor1B = 18
    Motor1E = 22

    Motor2A = 11
    Motor2B = 13
    Motor2E = 15

    GPIO.setup(Motor1A, GPIO.OUT)
    GPIO.setup(Motor1B, GPIO.OUT)
    GPIO.setup(Motor1E, GPIO.OUT)

    GPIO.setup(Motor2A, GPIO.OUT)
    GPIO.setup(Motor2B, GPIO.OUT)
    GPIO.setup(Motor2E, GPIO.OUT)



@app.route("/left")
def left():
    setup()
    GPIO.output(Motor1A, GPIO.HIGH)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor1E, GPIO.HIGH)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor2B, GPIO.HIGH)
    GPIO.output(Motor2E, GPIO.HIGH)
    sleep(1)
    stop()
    return redirect("/")



@app.route("/right")
def right():
    setup()
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.HIGH)
    GPIO.output(Motor1E, GPIO.HIGH)
    GPIO.output(Motor2A, GPIO.HIGH)
    GPIO.output(Motor2B, GPIO.LOW)
    GPIO.output(Motor2E, GPIO.HIGH)
    sleep(1)
    stop()
    return redirect("/")




@app.route("/forward")
def forward():
    setup()
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.HIGH)
    GPIO.output(Motor1E, GPIO.HIGH)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor2B, GPIO.HIGH)
    GPIO.output(Motor2E, GPIO.HIGH)
    sleep(1)
    stop()
    return redirect("/")


@app.route("/backward")
def backward():
    setup()
    GPIO.output(Motor1A, GPIO.HIGH)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor1E, GPIO.HIGH)
    GPIO.output(Motor2A, GPIO.HIGH)
    GPIO.output(Motor2B, GPIO.LOW)
    GPIO.output(Motor2E, GPIO.HIGH)
    sleep(1)
    stop()
    return redirect("/")


def stop():
    GPIO.output(Motor1E, GPIO.LOW)
    GPIO.output(Motor2E, GPIO.LOW)
    GPIO.cleanup()


if __name__ == "__main__":

        ap = argparse.ArgumentParser()
        ap.add_argument("-f","--frame-count", type=int, default=32, help=
        " number of frames used to constfuct the background model")
        ap.add_argument("-c", "--confidence", type=float, default=0.2, help=
        "minimum porbability  to  filter weak detection")

        args = vars(ap.parse_args())
        t = threading.Thread(target=detect_motion, args=(args["frame_count"],))
        t.daemon = True
        t.start()

        app.run(host="0.0.0.0", port=8000, debug=True,threaded=True, use_reloader=False)


vs.stop()






