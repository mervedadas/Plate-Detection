import cv2
import numpy as np
from foregroundOfImage import foreground
from adaptiveCanny import auto_canny

img = cv2.imread("images/real_car.jpg",cv2.IMREAD_COLOR)
img =cv2.resize(img, (720, 640))
cv2.imshow("original", img)
img = foreground(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(gray,5)
# edged = cv2.Canny(median,30,200)
edged= auto_canny(median)
# cv2.imshow("canny", edged)
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[0:10]
font = cv2.FONT_HERSHEY_COMPLEX
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (255,0,0), 3)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 4:
        screenCnt = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

cv2.imwrite("plate.jpg",Cropped)
cv2.imshow("cropped",Cropped)

cv2.imshow("final",img)
cv2.waitKey(0)
cv2.destroyAllWindows()