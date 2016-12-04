import cv2
import json

import numpy as np
import pyscreenshot as ImageGrab

from iolistener import KeyBoardEventListener, MouseClickEventListener

print('--- Endless Run Neural Network Approach ---')
print('[X] Press "q" to quit')
print('[!] Initializing...')

# Settings
with open('config.json', 'r') as f:
    SETTINGS = json.load(f)

print('[!] Ensure that the game window is initialized before proceeding')
print('[!] Please click the top left and bottom right of the game window, and leave some margins')

# Our IO listeners
keyevents = KeyBoardEventListener()
mouseevents = MouseClickEventListener()
keyevents.start()
mouseevents.start()

# Wait until user specifies windows dimensions
while len(mouseevents.clicked_positions) < 2:
    pass

''' ROI window calibration '''
# ROI for game window
ROI_GAME = [i for sub in mouseevents.clicked_positions for i in sub]

# Grab screenshot of image
img = ImageGrab.grab(bbox=tuple(ROI_GAME))
img = np.array(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Grayscale, blur, and apply otsu for dynamic thresholding
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contour of the image
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Keep only largest contour and crop image to the ROI
# Goal is to get the ROI
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(cnts)

# Uncomment below to debug
# img_roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
# cv2.imshow('img', img_roi)
# cv2.waitKey(0)
raw_input('[!] Calibration complete, input anything to continue')
print('[!] Press "q" to quit')
keyevents.end = False

# Rescale image to 85 x 145 (width, height)
while not keyevents.end:
    img = ImageGrab.grab(bbox=tuple(ROI_GAME))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    img = cv2.resize(img_roi, (SETTINGS['scaledx'], SETTINGS['scaledy']), interpolation=cv2.INTER_CUBIC)

print('[X] Quitted')

