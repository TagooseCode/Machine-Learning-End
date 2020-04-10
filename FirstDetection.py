from imageai.Detection import ObjectDetection   # IMPORT ImageAI object detection class
import os   # IMPORT Python os class

execution_path = os.getcwd()

detector = ObjectDetection()    # DEFINE object detection variable
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    ## !!! 'resnet50_coco_best_v2.0.1.h5' is the RetinaNet model file used for object detection
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
    ## !!! IMAGE FILE MUST BE CALLED 'image.jpg'

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )