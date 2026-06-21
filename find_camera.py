import cv2

# Try camera numbers 0 through 3 and report which ones actually work
for i in range(4):
    cap = cv2.VideoCapture(i)
    ok, frame = cap.read()
    if ok:
        print(f"Camera {i}: WORKS  (frame size {frame.shape[1]}x{frame.shape[0]})")
    else:
        print(f"Camera {i}: nothing")
    cap.release()