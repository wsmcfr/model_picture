import cv2
print('OpenCV版本:', cv2.__version__)
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if ret:
            print(f'摄像头 {i}: {w}x{h} @ {fps}fps, 读取成功, 帧尺寸={frame.shape}')
        else:
            print(f'摄像头 {i}: {w}x{h}, 打开但无法读取帧')
        cap.release()
    else:
        print(f'摄像头 {i}: 无法打开')
