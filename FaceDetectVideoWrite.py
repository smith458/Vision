import cv2
import datetime

#Face finding cascade selection
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#Capture video: 0 - internal cam. 1 - webcam.
cap = cv2.VideoCapture(0)

#Setup Writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')

#File name and output setup
dt = str(datetime.datetime.now())
dtFile = dt.replace(' ', '_').replace('.', '-').replace(':', '-')
outputFile = 'Video/' + dtFile + '.avi'
out = cv2.VideoWriter(outputFile, fourcc, 20.0, (640,480))

while (True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
    )

    print("Found {0} faces".format(len(faces)))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('video', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()