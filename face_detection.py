import sys, cv2


cascade_path = 'face.xml'


def cascade_detect(cascade, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(
        gray_image,
        scaleFactor = 1.15,
        minNeighbors = 5,
        minSize = (120, 120),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
 
    
def detections_draw(image, detections):
    for (x, y, w, h) in detections:
        print "({0}, {1}, {2}, {3})".format(x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

 
def frame_face(image_path, result_path):
    cascade = cv2.CascadeClassifier(cascade_path)
    image = cv2.imread(image_path)
    if image is None:
        print "ERROR: Image did not load."
        return 2

    detections = cascade_detect(cascade, image)
    detections_draw(image, detections)
 
    cv2.imwrite(result_path, image)

