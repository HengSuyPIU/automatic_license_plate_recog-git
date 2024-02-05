from ultralytics import YOLO
import cv2
#https://github.com/abewley/sort.git
from sort.sort import *
from util import get_car, read_license_plate, write_csv

# save information
results = {}

# object tracker from sort.sort
mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt') # detect cars

license_plates_detector = YOLO('./best.pt') # detect license plates

# load video
cap = cv2.VideoCapture('./car_clip1.mp4')

# index for vehicle detection with pre-trained coco_model data
# raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles: # check if object detected is vehicle
                detections_.append([x1, y1, x2, y2, score])
                
        #track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        
        # detect license plate
        license_plates = license_plates_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # assign license plate car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            
            # apply grayscale conversion and crop in for better view
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            license_plate_crop_threshold = cv2.adaptiveThreshold(license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 15)
            # _, license_plate_crop_threshold = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            
            # read license plate from the threshold crop
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_threshold)
            
            #explanation:
            # if license_plate is detected:
            # for every frame(frame_nmr), save all information for related to cars(car_id)
            # and for each car('car'), save the information about that car(xcar1, ycar1...)
            # and for each car, also save all information about the license plate
            if license_plate_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]}, 
                                              'license_plate': {'bbox':[x1, y1, x2, y2],
                                                  'text': license_plate_text,
                                                  'bbox_score': score,
                                                  'text_score': license_plate_text_score}}
            
            
# write result to csv file
write_csv(results, './test.csv')

# cv2.imshow('original crop', license_plate_crop)
# cv2.imshow('threshold crop', license_plate_crop_threshold)

# cv2.waitKey(0)          