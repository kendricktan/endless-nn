import numpy as np
import cv2
import pyscreenshot as ImageGrab
from curtsies import Input

while True:
    with Input(keynames='curtsies') as input_generator:
        # Hard coded, basically chrome window snapped to the right
        # half of the screen
        img = ImageGrab.grab(bbox=(150, 125, 515, 750))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Dynamically get the ROI
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.destroyAllWindows()
print('quitted')