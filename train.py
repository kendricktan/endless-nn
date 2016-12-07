import cv2
import json
import os
import pickle
import time

import numpy as np
from neat import nn, population
from neat.config import Config
from pymouse import PyMouse

import screeny
from iolistener import KeyBoardEventListener, MouseClickEventListener

print('--- Endless Run Neural Network Approach ---')
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

# Where to click to jump
CLICK_JUMP_LOCATION_X = ROI_GAME[0] + (ROI_GAME[2] / 2)
CLICK_JUMP_LOCATION_Y = ROI_GAME[1] + (ROI_GAME[3] / 2)

# How many runs per network
RUNS_PER_NET = 5

# Our scales for converting the image into NN input
SCALEX = 480 / SETTINGS['scaledx']
SCALEY = 840 / SETTINGS['scaledy']


def eval_genome(genomes):
    '''Fitness function for the GE'''
    for g in genomes:
        # visualize.draw_net(g, view=True, fmt='png')
        net = nn.create_feed_forward_phenotype(g)
        fitnesses = []
        for i in range(RUNS_PER_NET):
            start_time = time.time()
            while not keyevents.end:
                img = screeny.screenshot(region=tuple(ROI_GAME))
                img = np.array(img)
                img = cv2.resize(img, (
                481, 841))  # Resize to a fixed size that we know works well with the current scalex and scaley (8 x 15)

                # Platform + coin thresholding
                # Bitwise OR to get better view of platform
                # Blur to reduce noise
                # Morphological transformation to reduce noise
                masked_platform = cv2.inRange(img, LOWER_RGB_PLATFORM, UPPER_RGB_PLATFORM)
                masked_coin = cv2.inRange(img, LOWER_RGB_COIN, UPPER_RGB_COIN)
                masked_platform = cv2.bitwise_or(masked_platform, masked_coin)
                masked_platform = cv2.medianBlur(masked_platform, 3)
                masked_platform = cv2.morphologyEx(masked_platform, cv2.MORPH_CLOSE, KERNEL)

                # Input to NN
                neat_input = np.zeros((SETTINGS['scaledy'], SETTINGS['scaledx']))
                y_in = 0
                x_in = 0
                for x in range(0, img.shape[1] - SCALEX, SCALEX):
                    for y in range(0, img.shape[0] - SCALEY, SCALEY):
                        cv2.rectangle(img, (x, y), (x + SCALEX, y + SCALEY), (0, 255, 0), 2)
                        cur_img_roi = masked_platform[y:y + SCALEY, x:x + SCALEX]
                        cur_img_roi = cur_img_roi.flatten()

                        # If there's a decent amount of white in it, consider it a playform
                        if len(cur_img_roi[cur_img_roi == 255]) > 50:
                            neat_input[y_in, x_in] = 1

                        y_in += 1
                    x_in += 1
                    y_in = 0

                # NEAT evaluation takes place here
                inputs = neat_input.flatten()
                output = net.serial_activate(inputs)
                if output[0] > 0.5:
                    mousehandler.click(CLICK_JUMP_LOCATION_X, CLICK_JUMP_LOCATION_Y, 1)

                # Check if we lost
                masked_fb_button = cv2.inRange(img, LOWER_RGB_PLAY_BUTTON, UPPER_RGB_PLAY_BUTTON)

                if np.count_nonzero(masked_fb_button) > 0:
                    fitness = time.time() - start_time
                    fitness = round(fitness, 2)

                    # Impossible to have such a low fitness
                    if (fitness > 0.5):
                        fitnesses.append(fitness)
                        print('[-] Lost, seconds elapsed: {}'.format(fitness))

                    mousehandler.click(SHOP_BUTTON_POSITION_X, SHOP_BUTTON_POSITION_Y, 1)
                    mousehandler.click(PLAY_BUTTON_POSITION_X, PLAY_BUTTON_POSITION_Y, 1)

                    # Delay for the game to resume
                    time.sleep(1)

                    # Check for the shop replay button (occurs @ 2k coins)
                    img = screeny.screenshot(region=tuple(ROI_GAME))
                    img = np.array(img)
                    masked_shop_replay = cv2.inRange(img, LOWER_RGB_SHOP_BUTTON, UPPER_RGB_SHOP_BUTTON)
                    masked_shop_replay = cv2.erode(masked_shop_replay, KERNEL)

                    if np.count_nonzero(masked_shop_replay) > 15:
                        mousehandler.click(SHOP_BUTTON_POSITION_X, SHOP_BUTTON_POSITION_Y, 1)
                        time.sleep(1)

                    break

        # Genome's fitness is the worst performance across all nets
        g.fitness = min(fitnesses)


# Magic happens here
local_dir = os.path.dirname(__file__)
nn_config = Config(os.path.join(local_dir, 'neat_config'))
pop = population.Population(nn_config)
pop.run(eval_genome, 2000)

print('Number of evaluations: {0}'.format(pop.total_evaluations))
best_genome = pop.statistics.best_genome()
with open('nn_winner_genome', 'wb') as f:
    pickle.dump(best_genome, f)

print('[X] Quitted')
