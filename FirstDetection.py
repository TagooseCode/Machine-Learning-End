#https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
from imageai.Detection import ObjectDetection   # IMPORT ImageAI object detection class
import os   # IMPORT Python os class
import cv2

def determine(eachObject, centerx, centery): # (x1, y1, x2, y2). x1 and y1 refers to the lowerleft corner 
    #and x2 and y2 refers to the upperright corner.
    if (eachObject["box_points"][0] <= centerx <= eachObject["box_points"][2]):
        if (eachObject["box_points"][1] <= centery <= eachObject["box_points"][3]):
            return True
    return False


im = cv2.imread('image.jpg')
h, w, c = im.shape

centerx = w/2
centery = h/2

execution_path = os.getcwd()

detector = ObjectDetection()    # DEFINE object detection variable
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    ## !!! 'resnet50_coco_best_v2.0.1.h5' is the RetinaNet model file used for object detection
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
    ## !!! IMAGE FILE MUST BE CALLED 'image.jpg'

finallist = [x for x in detections if determine(x,centerx,centery)]

if (len(finallist) > 0):
    left = finallist[0]["box_points"][0]
    top = finallist[0]["box_points"][1]
    right = finallist[0]["box_points"][2]
    bottom = finallist[0]["box_points"][3]
    im1 = im[top:bottom, left:right]
    cv2.imwrite('new.jpg',im1)

#only try saving URL to database after you've cropped the photo

