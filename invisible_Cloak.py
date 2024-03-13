import numpy as np
import cv2
import time

cap = cv2.VideoCapture(1)

time.sleep(3)

background = 0

for i in range(60):
    ret, background = cap.read()

background = np.flip(background, axis=1)

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Adjusting HSV range for red color detection
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Adjusting HSV range for brown color detection
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([30, 255, 255])
    mask3 = cv2.inRange(hsv, lower_brown, upper_brown)

    # Combining masks for red and brown color detection
    mask_combined = mask1 + mask2 + mask3

    # Morphological operations
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=1)
    mask_inv = cv2.bitwise_not(mask_combined)

    # Generating final output
    res1 = cv2.bitwise_and(background, background, mask=mask_combined)
    res2 = cv2.bitwise_and(img, img, mask=mask_inv)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow('Invisible Cloak', final_output)
    k = cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
