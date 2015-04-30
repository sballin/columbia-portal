import sys 
import cv2
import subprocess
import textwrap
import pygame
import picamera
import picamera.array
import numpy as np


cascade = cv2.CascadeClassifier('classifiers/opencv_frontalface.xml')


def cascade_detect(cascade, image, minSide):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(
        gray_image,
        scaleFactor = 1.15,
        minNeighbors = 5,
        minSize = (minSide, minSide),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
 

def draw_sunglasses(image, detections):
    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, int(1.3*y)), (x + w, int(1.65*y)), (0, 0, 0), -1)
    if len(detections) == 0:
        cv2.putText(image, 'No faces found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
    return image

    
def draw_thoughts(image, detections):
    for (x, y, w, h) in detections:
        text = subprocess.check_output(['fortune', '-sa']) 
        text = textwrap.fill(text, 25)
        lines = text.split('\n')
        line_number = 0
        cv2.rectangle(image, (x + w, y - 30), (x + w + 400, y + len(lines)*30), (0, 0, 0), -1)
        #cv2.ellipse(image, (x, y), (w, h), 0, 0, 180)#, ['black', 2]) 
        for line in lines:
            cv2.putText(image, line, (x + w, y + line_number*30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
            line_number += 1
    if len(detections) == 0:
        cv2.putText(image, 'No faces found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
    return image


def draw_frames(image, detections):
    for (x, y, w, h) in detections:
        print "({0}, {1}, {2}, {3})".format(x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 

image_size = (800, 600)
#img = np.ones(img_size)*255

pygame.init()
screen = pygame.display.set_mode(image_size, pygame.FULLSCREEN)

camera = picamera.PiCamera()
camera.resolution = image_size
with picamera.array.PiRGBArray(camera) as stream:
    while True:
        camera.capture(stream, 'rgb', use_video_port=True)
        image = stream.array
        detections = cascade_detect(cascade, image, 70)
        image = draw_sunglasses(image, detections)
        surf = pygame.surfarray.make_surface(image)
        surf = pygame.transform.rotate(surf, -90)
        surf = pygame.transform.flip(surf, 1, 0)
        screen.blit(surf, (0,0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    break
        stream.seek(0)
        stream.truncate()

camera.close()

