import cv2
import json
import sys
import time

import numpy as np
import pandas as pd
from pymouse import PyMouse
from sklearn.neural_network import MLPClassifier

import screeny
from iolistener import KeyBoardEventListener, MouseClickEventListener

# To collect data, run `python app.py collect`
# Are we collecting data or nah
IS_COLLECT = 'collect' in str(sys.argv)

print('--- Endless Run Neural Network Approach ---')
if IS_COLLECT:
    print('[!] Collect mode...')
else:
    print('[!] Autonomous mode...')
    INPUT = []
    OUTPUT = []

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 16), random_state=1)

    DATA = ['jump_data.csv', 'normal_data.csv']
    for fi in DATA:
        with open(fi, 'r') as f:
            df = pd.read_csv(f)

            # Read into dataframes
            for i in df.index:
                # Can't turn string to numpy array easily for some reason?
                INPUT.append(np.fromstring(df['input'][i][2:-1].replace('\n', '').replace('.', ''), sep=' '))
                OUTPUT.append(int(df['output'][i][1]))

    INPUT = np.array(INPUT)
    OUTPUT = np.array(OUTPUT)

    clf.fit(INPUT, OUTPUT)


print('[X] Press "q" to quit')
print('[!] Initializing...')

# Settings
with open('settings.json', 'r') as f:
    SETTINGS = json.load(f)

print('[!] Ensure that the game window is initialized before proceeding')
print('[!] Please click the top left and bottom right of the game window, and leave some margins')

# Our IO event handlers/listeners
mousehandler = PyMouse()
keyevents = KeyBoardEventListener()
mouseevents = MouseClickEventListener()
keyevents.start()
mouseevents.start()

# Wait until user specifies windows dimensions
while len(mouseevents.clicked_positions) < 2:
    pass

''' ROI window calibration '''
# ROI for game window
ROI_GAME = list(
    mouseevents.clicked_positions[0] + tuple(
        np.subtract(mouseevents.clicked_positions[1], mouseevents.clicked_positions[0])
    )
)

# Grab screenshot of image
img = screeny.screenshot(region=tuple(ROI_GAME))
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
ROI_GAME = [ROI_GAME[0] + roi_x, ROI_GAME[1] + roi_y, roi_w, roi_h]

# Uncomment below to debug
# cv2.waitKey(0)
print('[!] Calibration complete')
print('[!] Press "q" to quit')
keyevents.end = False

# Rescale image to 85 x 145 (width, height)
# Thresholding it on RGB should be fine
# since it's static colors
# Our upper and lower bound for thresholding
LOWER_RGB_PLATFORM = np.array(SETTINGS['platformmin_rgb'])
UPPER_RGB_PLATFORM = np.array(SETTINGS['platformmax_rgb'])
LOWER_RGB_COIN = np.array(SETTINGS['coinmin_rgb'])
UPPER_RGB_COIN = np.array(SETTINGS['coinmax_rgb'])
LOWER_RGB_PLAYER = np.array(SETTINGS['playermin_rgb'])
UPPER_RGB_PLAYER = np.array(SETTINGS['playermax_rgb'])
LOWER_RGB_PLAY_BUTTON = np.array(SETTINGS['playagain_min_rgb'])
UPPER_RGB_PLAY_BUTTON = np.array(SETTINGS['playagain_max_rgb'])
LOWER_RGB_SHOP_BUTTON = np.array(SETTINGS['shopbtn_min_rgb'])
UPPER_RGB_SHOP_BUTTON = np.array(SETTINGS['shopbtn_max_rgb'])

KERNEL = np.ones((5, 5), np.uint8)

# Play again button position
# assuming its 4.25% from the bottom
PLAY_BUTTON_POSITION_Y = ROI_GAME[3] - (ROI_GAME[3] * 4.25 / 100)
PLAY_BUTTON_POSITION_X = ROI_GAME[2] / 2
PLAY_BUTTON_POSITION_Y += ROI_GAME[1]
PLAY_BUTTON_POSITION_X += ROI_GAME[0]

# Shop replay button (occurs on 2k coins)
SHOP_BUTTON_POSITION_Y = ROI_GAME[3] - (ROI_GAME[3] * 14.5 / 100)
SHOP_BUTTON_POSITION_X = ROI_GAME[2] / 2
SHOP_BUTTON_POSITION_Y += ROI_GAME[1]
SHOP_BUTTON_POSITION_X += ROI_GAME[0]

# Our scales for converting the image into NN input
SCALEX = 480 / SETTINGS['scaledx']
SCALEY = 840 / SETTINGS['scaledy']

IN_TOTAL = []
OUT_TOTAL = []

if IS_COLLECT:
    cv2.imshow('img', img)
    cv2.waitKey(1)

time.sleep(3)
print('[!] Click anywhere to continue')
mouseevents.clicked = False
while not mouseevents.clicked:
    pass

print('[!] Starting...')
start_time = time.time()
while not keyevents.end:
    mouseevents.clicked = False

    img = screeny.screenshot(region=tuple(ROI_GAME))
    img = np.array(img)
    img = cv2.resize(img, (
        481,
        841)
                     )  # Resize to a fixed size that we know works well with the current scalex and scaley (8 x 15)

    # Platform + coin thresholding
    # Bitwise OR to get better view of platform
    # Blur to reduce noise
    # Morphological transformation to reduce noise
    masked_platform = cv2.inRange(img, LOWER_RGB_PLATFORM, UPPER_RGB_PLATFORM)
    masked_coin = cv2.inRange(img, LOWER_RGB_COIN, UPPER_RGB_COIN)
    masked_platform = cv2.bitwise_or(masked_platform, masked_coin)
    masked_platform = cv2.medianBlur(masked_platform, 3)
    masked_platform = cv2.morphologyEx(masked_platform, cv2.MORPH_CLOSE, KERNEL)

    # Input to Artifical Neural Network
    # Only want to feed it 3 tiles in front of the player
    ann_input = np.zeros((3, SETTINGS['scaledx']))

    # Masking player (Assuming it's the default player)
    # Get largest contour (most likely to be player)
    masked_player = cv2.inRange(img, LOWER_RGB_PLAYER, UPPER_RGB_PLAYER)
    masked_player = cv2.morphologyEx(masked_player, cv2.MORPH_OPEN, KERNEL)
    cnts, _ = cv2.findContours(masked_player.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        p_x, p_y, p_w, p_h = cv2.boundingRect(cnts)

        y_in = 0
        x_in = 0
        for y in range(0, img.shape[0] - SCALEY, SCALEY):
            # If they're not 2 grids in front, ignore
            if not ((p_y + (p_h * 3)) > y and (p_y + (p_h)) < (y + SCALEY)):
                continue

            for x in range(0, img.shape[1] - SCALEX, SCALEX):
                cv2.rectangle(img, (x, y), (x + SCALEX, y + SCALEY), (0, 0, 255), 2)
                cur_img_roi = masked_platform[y:y + SCALEY, x:x + SCALEX]
                cur_img_roi = cur_img_roi.flatten()

                # If there's a decent amount of white in it, consider it a playform
                if len(cur_img_roi[cur_img_roi == 255]) > len(cur_img_roi) / 4:
                    ann_input[y_in, x_in] = 1
                    cv2.rectangle(img, (x, y), (x + SCALEX, y + SCALEY), (0, 255, 0), 2)

                # Highlight player
                if y < p_y + p_h and y + SCALEY > p_y:
                    if x < p_x + p_w and x + SCALEX > p_x:
                        ann_input[y_in, x_in] = 1
                        cv2.rectangle(img, (x, y), (x + SCALEX, y + SCALEY), (255, 0, 0), 2)

                x_in += 1

            x_in = 0
            y_in += 1

            if (y_in) > 2:
                break

    except Exception as e:
        print("[E] Error: {}".format(e))

    if IS_COLLECT:
        output = []
        if mouseevents.clicked:
            output.append(1)

        else:
            output.append(0)

        IN_TOTAL.append(ann_input.flatten())
        OUT_TOTAL.append(output)

        cv2.imshow('img', img)
        cv2.waitKey(1)

    else:
        if clf.predict([ann_input.flatten()])[0] == 1:
            mousehandler.click(SHOP_BUTTON_POSITION_X, SHOP_BUTTON_POSITION_Y)

if IS_COLLECT:
    print('[S] Saving data...')
    raw_data = {
        'input': IN_TOTAL,
        'output': OUT_TOTAL
    }

    df = pd.DataFrame(raw_data, columns=['input', 'output'])
    with open('data.csv', 'w') as f:
        df.to_csv(f)

print('[X] Quitted')
