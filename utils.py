import numpy as np
from scipy import signal

def preprocess_image(image):
    return np.float32(image) / 127.5 - 1.0

def preprocess_mask(mask):
    w_k = np.ones((10,6)) 
    msk = signal.convolve2d(mask.astype(np.float32), w_k, 'same')
    msk = 1 - (msk==0)
    msk = np.expand_dims(msk, axis=-1)
    return np.float32(msk)

def postprocess_output(output):
    x = np.squeeze(output, axis=0)
    x = np.transpose(x, [1, 2, 0])
    x = (x + 1.0) / 2.0 * 255.0
    x = np.clip(x, 0.0, 255.0)
    return np.uint8(x)
