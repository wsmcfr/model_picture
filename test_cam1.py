import cv2
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print('opened:', cap.isOpened())
ret, frame = cap.read()
print('read:', ret, frame.shape if ret else 'fail')
cap.release()

# 再试DirectShow后端
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
ret2, frame2 = cap2.read()
print('DSHOW read:', ret2, frame2.shape if ret2 else 'fail')
cap2.release()
