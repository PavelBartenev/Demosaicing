from PIL import Image
import numpy as np
from demosaicing import Demosaicing
from gradients import get_psnr

image = Image.open("data/CFA.bmp")

reconstructor = Demosaicing(image)
reconstructor.process()
processed_image = reconstructor.demosaiced_image

Image.fromarray(processed_image).save('processed.png')

original_image = np.array(Image.open("data/Original.bmp"))
psnr = get_psnr(original_image, processed_image)
time_taken = reconstructor.time_taken
time_per_mp = time_taken / (np.prod(reconstructor.black_white_image.shape) / 10**6)

with open('process_info.txt', 'w') as file:
    file.write("Time taken per MP: {:.3f} s \n\nPSNR: {:.3f}".format(time_per_mp, psnr))