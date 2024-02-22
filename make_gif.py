# Script with some functions to merge images into gifs

import imageio
import tqdm
from glob import glob


def write_gif(files, output, fps=1):
    with imageio.get_writer(output, mode='I', fps=fps) as writer:
        for f in tqdm.tqdm(files):
            image = imageio.imread(f)
            writer.append_data(image)

# Get all figures
files = glob('/path/to/files/frame*.png')

# Merge figures into a GIF
print('Merging into GIF...')
write_gif(files, 'movie.gif')