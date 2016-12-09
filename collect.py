import json
import time

import pandas as pd

from eyes import Eyes
from iolistener import KeyBoardEventListener, MouseClickEventListener
from ops import get_roi_from_mouse, get_new_filename

print('[X] Press "q" to quit')
print('[!] Initializing...')
print('[!] Ensure that the game window is initialized before proceeding')
print('[!] Please click the top left and bottom right of the game window, and leave some margins')

# Our IO event listeners
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
eye = Eyes(ROI_GAME, SETTINGS, preview=True)
eye.tune_roi()

# Extra ubfi
print('[!] Calibration complete')
print('[!] Press "q" to quit')
print('[.] Please get ready')

# Time for user to get anything ready
print('[!] Press "c" to continue')
while not keyevents.pressed_c:
    pass

print('[!] Starting...')
keyevents.end = False

NN_INPUT = []
NN_OUTPUT = []

start_time = time.time()
while not keyevents.end:
    # To prevent data from being too bias to not clicking
    if time.time() - mouseevents.clicked_time > 0.08:
        mouseevents.clicked = False

    # 26 fps
    if time.time() - start_time < 0.038:
        continue

    # Append to total data
    NN_INPUT.append(eye.roi_to_grid().flatten())
    NN_OUTPUT.append(1 if mouseevents.clicked else 0)

    start_time = time.time()

# What filename to save as
file_outname = get_new_filename()

print('[S] Saving data...')
raw_data = {
    'input': NN_INPUT,
    'output': NN_OUTPUT
}
df = pd.DataFrame(raw_data, columns=['input', 'output'])
with open(file_outname, 'w') as f:
    df.to_csv(f)
print('[X] Finished')
