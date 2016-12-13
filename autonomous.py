import cPickle
import json
import time
import cv2

from pymouse import PyMouse

from eyes import Eyes
from iolistener import KeyBoardEventListener, MouseClickEventListener
from ops import get_roi_from_mouse

print('[X] Press "q" to quit')
print('[!] Initializing...')
print('[!] Ensure that the game window is initialized before proceeding')
print('[!] Please click the top left and bottom right of the game window, and leave some margins')

# Our IO event listeners
mouse = PyMouse()
keyevents = KeyBoardEventListener()
mouseevents = MouseClickEventListener()
keyevents.start()
mouseevents.start()

# Drawing stuff
font = cv2.FONT_HERSHEY_SIMPLEX

# Wait until user specifies windows dimensions
while len(mouseevents.clicked_positions) < 2:
    pass

# Load settings
with open('settings.json', 'r') as f:
    SETTINGS = json.load(f)

# ROI for game window
ROI_GAME = get_roi_from_mouse(mouseevents.clicked_positions)

# Our eyes for the game
eye = Eyes(ROI_GAME, SETTINGS)
eye.tune_roi()

# Extra
print('[!] Calibration complete')
print('[!] Loading model')

# Loading trained model
with open('model.pkl', 'rb') as fid:
    clf = cPickle.load(fid)

print('[!] Loaded model')
print('[!] Press "q" to quit')
print('[.] Please get ready')
print('[!] Press "c" to continue')
while not keyevents.pressed_c:
    pass

print('[!] Starting...')
keyevents.end = False

# Time used to display 'jump'
start_time = time.time()
start_time -= 32767
while not keyevents.end:
    ann_input = eye.roi_to_grid().flatten()
    img_preview = eye.img_preview

    if clf.predict([ann_input])[0] == 1:
        start_time = time.time()
        mouse.click(eye._click_x, eye._click_y)
        time.sleep(0.06)

    # Shows text for 0.5 seconds
    if time.time() - start_time < 0.5:
        cv2.putText(img_preview, 'Jump!', (65, 65), font, 2, (0, 0, 0), 2)

    cv2.imshow('preview', img_preview)
    cv2.waitKey(1)

print('[X] Finished')
