import cv2 as cv

# Starts webcam
webcam = cv.VideoCapture(1)
# Color used to enhance classifier's detection
gray = cv.COLOR_BGR2GRAY

# Create variables to allocate pre-trained Cascade Classifiers
face_classifier = cv.CascadeClassifier("HaarCascades/haarcascade_frontalface_default.xml")
eye_classifier = cv.CascadeClassifier("HaarCascades/haarcascade_eye.xml")

while True:
    # Unpacks webcam.read(), boolean verifies VideoCapture's successfulness
    boolean, frame = webcam.read()

    # Changes frame's color to gray
    gray_frame = cv.cvtColor(frame, gray)

    # Uses face_classifier to detect faces in gray_frame
    face_detection = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.6, minNeighbors=4)

    for (x_axis, y_axis, width, height) in face_detection:

        # if x_axis > 0 and y_axis > 0:
        #     print('* PROCESS HALTED *')

        # Sets a rectangle on classified/detected faces
        cv.rectangle(frame, (x_axis, y_axis), (x_axis + width, y_axis + height), (0, 255, 0), 2)

        eye = frame[y_axis: y_axis + height, x_axis: x_axis + width]
        # Changes eye's color to gray
        gray_eye = cv.cvtColor(eye, gray)
        # Uses eye_classifier to detect eyes in gray_eye
        eye_detection = eye_classifier.detectMultiScale(gray_eye, scaleFactor=1.2, minNeighbors=4)

        for (ex_axis, ey_axis, ewidth, eheight) in eye_detection:
            # Sets a rectangle on classified/detected eyes
            cv.rectangle(eye, (ex_axis, ey_axis), (ex_axis + ewidth, ey_axis + eheight), (255, 0, 255), 2)

    # Displays frame
    cv.imshow("Face & Eyes Detection - Computer Vision", frame)

    # While loop breaks when 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

webcam.release()
cv.destroyAllWindows()
