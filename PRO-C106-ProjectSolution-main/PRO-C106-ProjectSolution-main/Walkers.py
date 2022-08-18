import cv2
bodyClassifier=cv2.CascadeClassifier("PRO-C106-ProjectSolution-main/haarcascade_fullbody.xml")
video=cv2.VideoCapture("PRO-C106-ProjectSolution-main/walking.avi")
while True:
    ret,image=video.read()
    greyscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    body=bodyClassifier.detectMultiScale(greyscale, 1.2, 3)
    for (x, y, w, h) in body:
        cv2.rectangle(image,(x, y), (x+w,y+h), (255,0,0), 2)
        cv2.imshow("pedestrian", image)
    if cv2.waitKey(1)==32:
        break
video.release()
cv2.destroyAllWindows()