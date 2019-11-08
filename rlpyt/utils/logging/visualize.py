import imageio
from PIL import Image
import numpy as np

def frames_to_gif(path, frames):
    frames = [Image.fromarray(np.transpose(frame, (1,2,0))) for frame in frames]
    imageio.mimsave(path, frames)

