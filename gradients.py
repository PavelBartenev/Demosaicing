import numpy as np


class Colors:
    red = 0
    green = 1
    blue = 2


def get_pixel_color(h, w):
    return w % 2 + h % 2


def calc_cardinal_grads(g1, b1, g2, b2, g3,
                        r1, g4, r2, g5, r3,
                        g6, b3, g7, b4, g8,
                        r4, g9, r5, g10, r6,
                        g11, b5, g12, b6, g13):
    gradN = abs(r2 - r5) + abs(g2 - g7) + abs(g4 - g9) / 2 + \
            abs(g5 - g10) / 2 + abs(b1 - b3) / 2 + abs(b2 - b4) / 2

    gradE = abs(b4 - b3) + abs(g8 - g7) + abs(g5 - g4) / 2 + \
            abs(g10 - g9) / 2 + abs(r3 - r2) / 2 + abs(r6 - r5) / 2

    gradS = abs(r5 - r2) + abs(g12 - g7) + abs(g9 - g4) / 2 + \
            abs(g10 - g5) / 2 + abs(b5 - b3) / 2 + abs(b6 - b4) / 2

    gradW = abs(b3 - b4) + abs(g6 - g7) + abs(g4 - g5) / 2 + \
            abs(g9 - g10) / 2 + abs(r1 - r2) / 2 + abs(r4 - r5) / 2

    return gradN, gradE, gradS, gradW


def calc_ordinal_grads(g1, b1, g2, b2, g3,
                       r1, g4, r2, g5, r3,
                       g6, b3, g7, b4, g8,
                       r4, g9, r5, g10, r6,
                       g11, b5, g12, b6, g13, color):
    if color == Colors.green:
        gradNE = abs(g5 - g9) + abs(g3 - g7) + abs(b2 - b3) + abs(r3 - r5)

        gradSE = abs(g10 - g4) + abs(g13 - g7) + abs(b6 - b3) + abs(r6 - r2)

        gradNW = abs(g4 - g10) + abs(g1 - g7) + abs(b1 - b4) + abs(r1 - r5)

        gradSW = abs(g9 - g5) + abs(g11 - g7) + abs(b5 - b4) + abs(r4 - r2)
    else:
        gradNE = abs(g5 - g9) + abs(g3 - g7) + (abs(r2 - b3) + abs(b4 - r5) + abs(b2 - r2) + abs(r3 - b4)) / 2

        gradSE = abs(g10 - g4) + abs(g13 - g7) + (abs(r5 - b3) + abs(b6 - r5) + abs(r6 - b4) + abs(r2 - b4)) / 2

        gradNW = abs(g4 - g10) + abs(g1 - g7) + (abs(r1 - b3) + abs(b3 - r5) + abs(b1 - r2) + abs(r2 - b4)) / 2

        gradSW = abs(g9 - g5) + abs(g11 - g7) + (abs(r4 - b3) + abs(b3 - r2) + abs(b5 - r5) + abs(r5 - b4)) / 2

    return gradNE, gradSE, gradNW, gradSW


def mean_colors_green_center(g1, b1, g2, b2, g3,
                             r1, g4, r2, g5, r3,
                             g6, b3, g7, b4, g8,
                             r4, g9, r5, g10, r6,
                             g11, b5, g12, b6, g13):
    return {'N': {'R': r2, 'G': (g2 + g7) / 2, 'B': (b1 + b2 + b3 + b4) / 4},
            'E': {'R': (r2 + r3 + r5 + r6) / 4, 'G': (g7 + g8) / 2, 'B': b4},
            'S': {'R': r5, 'G': (g7 + g12) / 2, 'B': (b3 + b4 + b5 + b6) / 4},
            'W': {'R': (r1 + r2 + r4 + r5) / 4, 'G': (g6 + g7) / 2, 'B': b3},
            'NE': {'R': (r2 + r3) / 2, 'G': g5, 'B': (b2 + b4) / 2},
            'SE': {'R': (r5 + r6) / 2, 'G': g10, 'B': (b4 + b6) / 2},
            'NW': {'R': (r1 + r2) / 2, 'G': g4, 'B': (b1 + b3) / 2},
            'SW': {'R': (r4 + r5) / 2, 'G': g9, 'B': (b3 + b5) / 2}}


def mean_colors_redblue_center(g1, b1, g2, b2, g3,
                        r1, g4, r2, g5, r3,
                        g6, b3, g7, b4, g8,
                        r4, g9, r5, g10, r6,
                        g11, b5, g12, b6, g13):
    return {'N': {'R': (g2 + g7) / 2, 'G': r2, 'B': (g4 + g5) / 2},
            'E': {'R': (g7 + g8) / 2, 'G': b4, 'B': (g5 + g10) / 2},
            'S': {'R': (g7 + g12) / 2, 'G': r5, 'B': (g9 + g10) / 2},
            'W': {'R': (g7 + g6) / 2, 'G': b3, 'B': (g4 + g9) / 2},
            'NE': {'R': (g7 + g3) / 2, 'G': (r2 + b2 + r3 + b4) / 4, 'B': g5},
            'SE': {'R': (g7 + g13) / 2, 'G': (b4 + b6 + r5 + r6) / 4, 'B': g10},
            'NW': {'R': (g7 + g1) / 2, 'G': (r1 + b1 + r2 + b3) / 4, 'B': g4},
            'SW': {'R': (g7 + g11) / 2, 'G': (b3 + b5 + r4 + r5) / 4, 'B': g9}}


def get_psnr(original, processed):
    MSE = 0
    for h in range(original.shape[0]):
        for w in range(original.shape[1]):
            r1, g1, b1 = original[h, w, :].tolist()
            r2, g2, b2 = processed[h, w, :].tolist()
            MSE += (0.299*(r2 - r1) + 0.587*(g2 - g1) + 0.114*(b2 - b1))**2

    MSE /= (original.shape[0] * original.shape[1])

    PSNR = 10 * np.log10(255**2 / MSE)

    return PSNR
