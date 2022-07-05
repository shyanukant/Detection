import cv2
car_classifier_file = cv2.CascadeClassifier('cars.xml')
pedestrian_classifier_file = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# capture with cam
webcam = cv2.VideoCapture('0')

# run loop untill frame's end.
while True:
    frame_read, frame = webcam.read()

    if frame_read:

        grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    car_coordinates = car_classifier_file.detectMultiScale(grayscaled)
    pedestrian_coordinates = pedestrian_classifier_file.detectMultiScale(grayscaled)
    # draw rectangle around object
    for (x, y, w , h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

    for (x, y, w, h) in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)

    cv2.imshow('Shyanukant Car Detector', frame)

    key = cv2.waitKey(1)
    
    if key == 81 or key == 113:
        break

webcam.release()

print("Code Run Successfully !!")