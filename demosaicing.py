import numpy as np
from tqdm import tqdm
import time
from gradients import calc_ordinal_grads, calc_cardinal_grads, Colors, get_pixel_color
from gradients import mean_colors_green_center, mean_colors_redblue_center


class Demosaicing:
    def __init__(self, image):
        self.black_white_image = np.array(image)

        self.window_size = 5
        self.window_half = self.window_size // 2

        self.height_start = self.window_size // 2
        self.height_end = self.black_white_image.shape[0] - self.window_size // 2

        self.width_start = self.window_size // 2
        self.width_end = self.black_white_image.shape[1] - self.window_size // 2

    def process_pixel(self, h, w):
        color = get_pixel_color(h, w)
        color_up = get_pixel_color(h + 1, w)

        window = 255 - np.array(self.black_white_image[h - self.window_half: h + self.window_half + 1,
                                                       w - self.window_half: w + self.window_half + 1]).flatten()

        window = window.tolist()

        g1, b1, g2, b2, g3, \
        r1, g4, r2, g5, r3, \
        g6, b3, g7, b4, g8, \
        r4, g9, r5, g10, r6, \
        g11, b5, g12, b6, g13 = window

        gradN, gradE, gradS, gradW = calc_cardinal_grads(*window)

        gradNE, gradSE, gradNW, gradSW = calc_ordinal_grads(*window, color)

        directions_grads = [gradN, gradE, gradS, gradW, gradNE, gradSE, gradNW, gradSW]
        directions_names = ['N', 'E', 'S', 'W', 'NE', 'SE', 'NW', 'SW']

        min_grad = min(directions_grads)
        max_grad = max(directions_grads)

        T = 1.5 * min_grad + 0.5 * (max_grad + min_grad)

        mean_colors = mean_colors_green_center(*window) if color == Colors.green \
            else mean_colors_redblue_center(*window)

        accepted_directions = [directions_names[i] for i in range(len(directions_grads)) if directions_grads[i] <= T]

        if not accepted_directions:
            return self.black_white_image[h, w]

        red_sum = green_sum = blue_sum = 0

        for direct in accepted_directions:
            green_sum += mean_colors[direct]['G']
            if color_up == Colors.red:
                red_sum += mean_colors[direct]['R']
                blue_sum += mean_colors[direct]['B']
            else:
                red_sum += mean_colors[direct]['B']
                blue_sum += mean_colors[direct]['R']

        cur_sum = (red_sum, green_sum, blue_sum)[color]

        cur_red = g7 + (red_sum - cur_sum) / len(accepted_directions)
        cur_green = g7 + (green_sum - cur_sum) / len(accepted_directions)
        cur_blue = g7 + (blue_sum - cur_sum) / len(accepted_directions)

        return np.array([cur_red, cur_green, cur_blue])

    def get_bayer_rgb(self):
        cfa_rgb = np.zeros((self.black_white_image.shape[0], self.black_white_image.shape[1], 3))
        for h in range(self.black_white_image.shape[0]):
            for w in range(self.black_white_image.shape[1]):
                cfa_rgb[h, w, get_pixel_color(h, w)] = 255 - self.black_white_image[h][w]

        return cfa_rgb

    def process(self):
        time_start = time.time()
        self.demosaiced_image = np.zeros((self.black_white_image.shape[0], self.black_white_image.shape[1], 3))

        for h in tqdm(range(self.height_start, self.height_end)):
            for w in range(self.width_start, self.width_end):
                self.demosaiced_image[h, w, :] = self.process_pixel(h, w)

        self.demosaiced_image[self.demosaiced_image > 255] = 255
        self.demosaiced_image[self.demosaiced_image < 0] = 0

        self.demosaiced_image = self.demosaiced_image.astype(np.uint8)
        time_end = time.time()

        self.time_taken = time_end - time_start

        return self.demosaiced_image
