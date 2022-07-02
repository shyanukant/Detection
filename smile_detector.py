import cv2
faceTrackerFile = cv2.CascadeClassifier('face_smile_detector/haarcascade_frontalface_default.xml')
smileTrackerFile = cv2.CascadeClassifier('face_smile_detector/haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    frame_read, frame = webcam.read()

    if not frame_read:

        break

    greyscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinate = faceTrackerFile.detectMultiScale(greyscaled)
    
    

    for (x, y, w, h) in face_coordinate:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 10), 5)
        #get the sub frame(using numpy N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        #change to grayscale
        face_grayscaled = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile_coordinate = smileTrackerFile.detectMultiScale(face_grayscaled, scaleFactor=1.7, minNeighbors=20 )

        # find all smiles in the face
        for (x_, y_, w_, h_) in smile_coordinate:

            #Draw a rectangle around the face
            cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 5)
        
        if len(smile_coordinate) > 0:
            cv2.putText(frame, 'Smile', (x, y+h+40), fontScale=2, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,color=(250,250,250))
    cv2.imshow('Smile Detector', frame)

    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()