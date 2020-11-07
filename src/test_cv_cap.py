import cv2

vcap = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")

while (True):
    ret, frame = vcap.read()
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)