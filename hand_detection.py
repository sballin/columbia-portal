import sys, cv2
import subprocess
import textwrap


cascade = cv2.CascadeClassifier('classifiers/hand-1.xml')


def cascade_detect(cascade, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(
        gray_image,
        scaleFactor = 1.15,
        minNeighbors = 5,
        minSize = (120, 120),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
 
    
def draw_bubbles(image_path, result_path):
    image = cv2.imread(image_path)
    if image is None:
        print "ERROR: Image did not load."
        return 2

    detections = cascade_detect(cascade, image)
 
    for (x, y, w, h) in detections:
        print "({0}, {1}, {2}, {3})".format(x, y, w, h)
        text = subprocess.check_output(['fortune', '-sa'])
        text = textwrap.fill(text, 25)
        print text
        cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
    cv2.imwrite(result_path, image)


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


#draw_bubbles('../test3.jpg', '../out3.jpg')
frame_faces('../test2.jpg', '../out2.jpg')
