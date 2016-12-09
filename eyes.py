import cv2

import numpy as np

import screeny


class Eyes:
    def __init__(self, _roi, _settings, ann_y=6, preview=False):
        '''
        args:
            _roi = [x, y, w, h] of the rough game window roi
            _settings = dict containing lower and upper bounds for thres
        '''
        self._roi = _roi
        self._settings = _settings
        self._ann_y = ann_y  # How many rows in front do we want to use as NN inputs?
        self._preview = preview
        self._kernel = np.ones((5, 5), np.uint8)  # Kernel to perform erosion/dilation

    def init_settings(self):
        '''
        Initializes the upper/lower bounds obtained from the settings dict
        as well as where to click
        
        All lower/upper bounds are in rgb format
        '''

        # Color thresholding
        self._lower_platform = np.array(self._settings['platformmin_rgb'])
        self._upper_platform = np.array(self._settings['platformmax_rgb'])

        self._lower_coin = np.array(self._settings['coinmin_rgb'])
        self._upper_coin = np.array(self._settings['coinmax_rgb'])

        self._lower_player = np.array(self._settings['playermin_rgb'])
        self._upper_player = np.array(self._settings['playermax_rgb'])

        # Grid system for the roi
        # grid_x_no depicts the number of grids on the horizontal axis
        # grid_y_no depicts the number of grids on the vertical axis
        self._grid_w = 480 / self._settings['grid_x_no']
        self._grid_h = 840 / self._settings['grid_y_no']

        # Where to click to jump
        self._click_x = self._roi[2] / 2.7
        self._click_x += self._roi[0]
        self._click_y = self._roi[3] - (self._roi[3] * 23.5 / 100)
        self._click_y += self._roi[1]

    def tune_roi(self):
        '''
        Given the supplied roi, try and crop out unecessary information
        '''
        # Grab screenshot of image
        img = screeny.screenshot(region=tuple(self._roi))
        img = np.array(img)

        # Grayscale, blur and dynamic thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours of the image
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Keep the largest contour and get contour's coordinates
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(cnt)
        self._roi = [self._roi[0] + roi_x, self._roi[1] + roi_y, roi_w, roi_h]

        img = screeny.screenshot(region=tuple(self._roi))
        img = np.array(img)

        if self._preview:
            cv2.imshow('preview', img)
            cv2.waitKey(1)

        # Once we tune the roi we'll automatically load settings
        self.init_settings()

    def roi_to_grid(self):
        '''
        Converts img to grids that'll be used as inputs for nn
        '''
        # Inputs for ANN
        ann_input = np.zeros((self._ann_y, self._settings['grid_x_no']))

        # Grabs screenshot and resizes img to a fixed size
        # that'll work well with the current grid_x_no and grid_y_no
        img = screeny.screenshot(region=tuple(self._roi))
        img = np.array(img)
        img = cv2.resize(img, (481, 841))
        img_preview = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

        # Platform + coin thresholding
        # Bitwise OR to fill in the blanks
        masked_platform = cv2.inRange(img, self._lower_platform, self._upper_platform)
        masked_coin = cv2.inRange(img, self._lower_coin, self._upper_coin)
        masked_platform = cv2.bitwise_or(masked_platform, masked_coin)
        masked_platform = cv2.medianBlur(masked_platform, 3)
        masked_platform = cv2.morphologyEx(masked_platform, cv2.MORPH_CLOSE, self._kernel)

        # Player thresholding
        # Also moprhological transformation to reduce noise
        masked_player = cv2.inRange(img, self._lower_player, self._upper_player)
        masked_player = cv2.morphologyEx(masked_player, cv2.MORPH_OPEN, self._kernel)
        cnts, _ = cv2.findContours(masked_player.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Might occasionally lose sight of player
        # So use try except
        try:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            for cnt in cnts:
                # Assume smallest player can ever get to is 500
                if cv2.contourArea(cnt) < 500:
                    break

                # Get x, y, w, h of player
                p_x, p_y, p_w, p_h = cv2.boundingRect(cnt)

                # Index count for ann_input
                y_in = 0
                x_in = 0

                for y in range(0, img.shape[0] - self._grid_h, self._grid_h):
                    # If they're not within a set range of player, ignore
                    if not ((p_y + (p_h * 4.5)) > y and (p_y + p_h) < (y + self._grid_h)):
                        continue

                    for x in range(0, img.shape[1] - self._grid_w, self._grid_w):
                        cv2.rectangle(img_preview, (x, y), (x + self._grid_w, y + self._grid_h), (0, 0, 255), 2)
                        cur_img_roi = masked_platform[y:y + self._grid_h, x:x + self._grid_w]
                        cur_img_roi = cur_img_roi.flatten()

                        # If there's a decent amount of white in the current thresholded image, consider it a platform
                        if len(cur_img_roi[cur_img_roi == 255]) > len(cur_img_roi) / 5:
                            ann_input[y_in, x_in] = 1
                            cv2.rectangle(img_preview, (x, y), (x + self._grid_w, y + self._grid_h), (0, 255, 0), 2)

                        # Highlight player
                        if y < p_y * 1.2 and y + self._grid_h > p_y * 0.8:
                            if x < p_x + p_w and x + self._grid_w > p_x:
                                ann_input[y_in, x_in] = 2
                                cv2.rectangle(img_preview, (x, y), (x + self._grid_w, y + self._grid_h), (255, 0, 0), 2)

                        x_in += 1
                    x_in = 0
                    y_in += 1

                    if y_in >= self._ann_y:
                        break

        except Exception as e:
            print('[E] {}'.format(e))

        # Show ya?
        if self._preview:
            cv2.imshow('preview', img_preview)
            cv2.waitKey(1)

        return ann_input
