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

# Uncomment below to debug
# img_roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
# cv2.imshow('img', img_roi)
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
# assuming its 6.25% from the bottom
PLAY_BUTTON_POSITION_Y = ROI_GAME[3] - (ROI_GAME[3] * 6.25 / 100)
PLAY_BUTTON_POSITION_X = ROI_GAME[2] / 2
PLAY_BUTTON_POSITION_Y += ROI_GAME[1]
PLAY_BUTTON_POSITION_X += ROI_GAME[0]

# Shop replay button (occurs on 2k coins)
SHOP_BUTTON_POSITION_Y = ROI_GAME[3] - (ROI_GAME[3] * 12.5 / 100)
SHOP_BUTTON_POSITION_X = ROI_GAME[2] / 2
SHOP_BUTTON_POSITION_Y += ROI_GAME[1]
SHOP_BUTTON_POSITION_X += ROI_GAME[0]

# Where to click to jump
CLICK_JUMP_LOCATION_X = ROI_GAME[0] + (ROI_GAME[2] / 2)
CLICK_JUMP_LOCATION_Y = ROI_GAME[1] + (ROI_GAME[3] / 2)

# How many runs per network
RUNS_PER_NET = 5


def eval_genome(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)
        fitnesses = []
        for i in range(RUNS_PER_NET):
            start_time = time.time()
            while not keyevents.end:
                img = screeny.screenshot(region=tuple(ROI_GAME))
                img = np.array(img)

                # Platform + coin thresholding
                # Bitwise OR to get better view of platform
                # Blur to reduce noise
                # Morphological transformation to reduce noise
                masked_platform = cv2.inRange(img, LOWER_RGB_PLATFORM, UPPER_RGB_PLATFORM)
                masked_coin = cv2.inRange(img, LOWER_RGB_COIN, UPPER_RGB_COIN)
                masked_platform = cv2.bitwise_or(masked_platform, masked_coin)
                masked_platform = cv2.medianBlur(masked_platform, 3)
                masked_platform = cv2.morphologyEx(masked_platform, cv2.MORPH_CLOSE, KERNEL)

                # Masking player (Assuming it's the default player)
                # masked_player = cv2.inRange(img, LOWER_RGB_PLAYER, UPPER_RGB_PLAYER)
                # masked_player = cv2.morphologyEx(masked_player, cv2.MORPH_OPEN, KERNEL)

                # Resize image (use this as input)
                masked_platform_resized = cv2.resize(masked_platform, (SETTINGS['scaledx'], SETTINGS['scaledy']),
                                                     interpolation=cv2.INTER_CUBIC)
                # masked_player_resized = cv2.resize(masked_player, (SETTINGS['scaledx'], SETTINGS['scaledy']), interpolation=cv2.INTER_CUBIC)

                # Combined image
                # masked_combined = cv2.bitwise_or(masked_platform_resized, masked_player_resized)

                # Jump
                # if random.randint(0, 1) == 1:
                #     mousehandler.click(CLICK_JUMP_LOCATION_X, CLICK_JUMP_LOCATION_Y, 1)

                # NEAT evaluation takes place here
                inputs = masked_platform_resized.flatten()
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
