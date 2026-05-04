import cv2
print("Testing DirectShow on camera 1...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
print("opened:", cap.isOpened())
ret, frame = cap.read()
print("read:", ret, frame.shape if ret else "fail")
cap.release()
