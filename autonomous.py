import cPickle
import json
import time

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

start_time = time.time()
while not keyevents.end:
    ann_input = eye.roi_to_grid().flatten()

    if clf.predict([ann_input])[0] == 1:
        mouse.click(eye._click_x, eye._click_y)
        time.sleep(0.06)

print('[X] Finished')
