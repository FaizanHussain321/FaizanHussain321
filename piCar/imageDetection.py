import numpy as np
import imutils
import cv2


class MotionDetector:


    def __init__(self, accumWeight=0.5):
        # stores the weight factor
        self.accumWeight = accumWeight

        # iniliazes the background model

        self.bg = None

    def update(self, image):

        # if the background model is non then it is initilised

        if self.bg is None:
            self.bg = image.copy().astype("float")

            # update the background model by accumulating the wieghted average

        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

        # performing movement in the video stream

    def detect(self, image, tVal=25):

        # calculate the absoulte difference between the background model

        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, tVal, 225, cv2.THRESH_BINARY)[1]

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # this wil find contours in the image and intiliase the minium and maximum
        # bounding box regions for motion

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        # if no contours were found then "none" shall be returned

        if len(cnts) == 0:
            return None

        # else loop over the contours

        for c in cnts:
            # calculate the bounding box of the contrours and use it to update the minium
            # and maximum bounding box regions

            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

            # otherwise return a tuple of the thresholded image along with a bounding box

        return (thresh, (minX, minY, maxX, maxY))









