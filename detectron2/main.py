from detector import *
import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

# img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27: # ESC pressed to quit
        print("Escape hit, closing...")
        break
    elif k%256 == 32: # SPACE pressed for capturing image, replace this with speech timing
        img_name = "captured.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        # img_counter += 1
        break

cam.release()

cv2.destroyAllWindows()


detector = Detector(model_type="IS")
detector.onImage("captured.png")