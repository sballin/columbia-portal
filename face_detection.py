import sys, cv2
import subprocess
import textwrap
import pygame
import picamera
import numpy as np
import io
import pickle


cascade = cv2.CascadeClassifier('classifiers/opencv_frontalface.xml')


def cascade_detect(cascade, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(
        gray_image,
        scaleFactor = 1.15,
        minNeighbors = 5,
        minSize = (70, 70),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
 
    
def draw_bubbles(image):
    detections = cascade_detect(cascade, image)
 
    for (x, y, w, h) in detections:
        print "({0}, {1}, {2}, {3})".format(x, y, w, h)
        text = subprocess.check_output(['fortune', '-sa']) # check for too many newlines
        text = textwrap.fill(text, 25)
        print text
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.ellipse(image, (x, y), (w, h), 0, 0, 180)#, ['black', 2]) 
    if len(detections) == 0:
        cv2.putText(image, 'No faces found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
    cv2.imwrite('../tests/test.jpg', image)
    return image


def draw_frames(image, detections):
    for (x, y, w, h) in detections:
        print "({0}, {1}, {2}, {3})".format(x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 

def frame_faces(image_path, result_path):
    image = cv2.imread(image_path)
    if image is None:
        print "ERROR: Image did not load."
        return 2

    detections = cascade_detect(cascade, image)
    draw_frames(image, detections)
 
    cv2.imwrite(result_path, image)


image_size = (800, 600)
#img = np.ones(img_size)*255

pygame.init()
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)

with picamera.PiCamera() as camera:
    try:
        camera.resolution = image_size
        stream = io.BytesIO()
    
        for foo in camera.capture_continuous(stream, format='png'):
            frame = np.fromstring(stream.getvalue(), dtype = np.uint8)
            stream.truncate()
            stream.seek(0)
            image = cv2.imdecode(frame, 1)
            image = draw_bubbles(image)
            image = pygame.image.load('../tests/test.jpg')
            #surf = pygame.surfarray.make_surface(image)    # note: makes colors bluish
            screen.blit(image, (0,0))
            pygame.display.flip()

    finally: camera.close()

