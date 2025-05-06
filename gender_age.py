'''
PyPower Projects
Detect Gender and Age using Artificial Intelligence
Modified for Live Webcam Detection
'''

import cv2 as cv

# Model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Age and gender lists
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Load networks
faceNet = cv.dnn.readNet(faceModel, faceProto)
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)

# Helper function to detect faces
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    return frameOpencvDnn, bboxes

# Start webcam
cap = cv.VideoCapture(0)
padding = 20

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                     max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227),
                                    (78.4263377603, 87.7689143744, 114.895847746),
                                    swapRB=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = GENDER_LIST[genderPreds[0].argmax()]

        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = AGE_LIST[agePreds[0].argmax()]

        label = "{},{}".format(gender, age)
        cv.rectangle(frameFace, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

    cv.imshow("Age-Gender Detection", frameFace)

    # Press 'q' to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
