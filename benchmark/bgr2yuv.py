import numpy as np
import torch

def bgr2ycbcr_npy(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ycbcr2rgb_npy(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bgr2ycbcr_torch(input_tensor, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]

        size : (B, C, H, W)
    '''
    if input_tensor.dtype != torch.uint8:
        image = input_tensor * 255.
    else:
        image = input_tensor.clone()

    if only_y:
        rgb_weights = [65.481, 128.553, 24.966]
    else:
        rgb_weights = [[65.481, 128.553, 24.966],
                       [-37.797, -74.203, 112.0],
                       [112.0, -93.786, -18.214]]
    rgb_weights = torch.tensor(rgb_weights, dtype=image.dtype, device=image.device)

    # RGB 채널을 곱하여 Y 계산
    if only_y:
        y = (rgb_weights[0] * image[:, 0:1, :, :] +
             rgb_weights[1] * image[:, 1:2, :, :] +
             rgb_weights[2] * image[:, 2:3, :, :])
    else:
        y = torch.stack([rgb_weights[0, 0] * image[:, 0, :, :] + rgb_weights[0, 1] * image[:, 1, :, :] + rgb_weights[0, 2] * image[:, 2, :, :],
                         rgb_weights[1, 0] * image[:, 0, :, :] + rgb_weights[1, 1] * image[:, 1, :, :] + rgb_weights[1, 2] * image[:, 2, :, :],
                         rgb_weights[2, 0] * image[:, 0, :, :] + rgb_weights[2, 1] * image[:, 1, :, :] + rgb_weights[2, 2] * image[:, 2, :, :]], dim=1)

    if input_tensor.dtype == torch.uint8:
        y = y.round()
    else:
        y /= 255.
    return y