import numpy as np
from skimage import draw
import tensorflow as tf

def center(arr, axis=0):
    arr_mus = tf.reduce_mean(arr, axis=axis)
    arr_centered = tf.math.subtract(arr, arr_mus)
    return arr_centered

def cov(arr, N, axis=0):
    arr_centered = center(arr, axis)
    cov = (tf.transpose(arr_centered) @ arr_centered) / (N-1)
    return cov

def standardize(arr, c=1e-7):
    import keras.backend as K
    mus = K.mean(arr, axis=0)
    stds = K.std(arr, axis=0)
#     stds[ stds==0 ] = 1 # prevent zero-devision
    stds += c # add small constant to prevend zero division
    return (arr - mus) / stds  # zero mean, std=1

def mnist(nsamples):
    import keras
    (x_train, _), (_, _) = keras.datasets.mnist.load_data()
    assert nsamples <= len(x_train)
    return tf.cast(x_train[:nsamples].reshape(nsamples, -1), dtype='float32')

def construct(nsamples, nrows, ncols, fill_type, dtype=np.float32):
    arr = np.zeros((nsamples, nrows*ncols), dtype=dtype)
    return np.apply_along_axis(globals()[fill_type], 1, arr, nrows, ncols)

def alpha(arr, *args, **kwargs):
    arr[:] = np.random.randint(0,255)
    return arr

def circle(arr, nrows, ncols):
    arr = arr.reshape(nrows, ncols)
    rr, cc = draw.disk(center=(nrows//2, ncols//2), 
            radius=np.random.randint(1, min(nrows//2, ncols//2)))
    arr[rr, cc] = 255
    return arr.flatten()

def ellipse(arr, nrows, ncols):
    arr = arr.reshape(nrows, ncols)
    rr, cc = draw.ellipse(r=nrows//2, c=ncols//2, 
            r_radius=np.random.randint(1, nrows//2),
            c_radius=np.random.randint(1, ncols//2),
            rotation=np.deg2rad(np.random.randint(180)))
    arr[rr, cc] = 255
    return arr.flatten()

if __name__ == "__main__":
    alphas = standardize(construct(10, 28, 28, "alpha"))
    circles = standardize(construct(10, 28, 28, "circle"))
    ellipses = standardize(construct(10, 28, 28, "ellipse"))
    mnist = standardize(mnist(10))
    print(alphas.shape, circles.shape, ellipses.shape, mnist.shape)
