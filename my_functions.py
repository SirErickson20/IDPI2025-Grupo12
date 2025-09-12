import numpy as np

MAT_YIQ = np.array([[0.299, 0.595716, 0.211456],
                    [0.587, -0.274453, -0.522591],
                    [0.114, -0.321263, 0.311135]])

def rgb2yiq(_im):
    _rgb = _im.reshape((-1,3))
    _yiq = _rgb @ MAT_YIQ
    _yiq = _yiq.reshape(_im.shape)
    return _yiq

def RGB_to_YIQ(rgb):
    yiq = np.zeros(rgb.shape)
    yiq[:,:,0] = 0.229*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    yiq[:,:,1] = 0.595716*rgb[:,:,0] - 0.274453*rgb[:,:,1] - 0.321263*rgb[:,:,2]
    yiq[:,:,2] = 0.211456*rgb[:,:,0] - 0.522591*rgb[:,:,1] + 0.311135*rgb[:,:,2]
    yiq[:,:,3] = rgb[:,:,3]
    return yiq

def yiq2rgb(_im):
    return (_im.reshape((-1, 3)) @ np.linalg.inv(MAT_YIQ)).reshape(_im.shape)

def rmse(im_1, im_2=0):
    return ((im_1 - im_2)**2).mean()**0.5

def change_intensity(_im, alpha=1, beta=1):
    """Change linearly luminance and intensity in yiq."""
    _yiq = rgb2yiq(_im) * np.array([alpha, beta, beta])[np.newaxis, np.newaxis, :]
    _rgb = yiq2rgb(_yiq)
    return _rgb