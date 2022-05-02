"""In this project, the input tensor is BGR255."""
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")

MEAN = [102.9801, 115.9465, 122.7717]


def show():
    cv2.waitKey()


def tensor2np(tensor):
    return tensor.detach().cpu().float().numpy()


def img_tensor2np_img(tensor):
    """BGR255 tensor [3, H, W] to BGR255 ndarray [H, W, 3]

    :param tensor: [C,H,W] tensor
    :return: ndarray [H, W, C]
    """
    img = tensor2np(tensor).transpose(1, 2, 0)
    img[:, :, 0] += MEAN[0]
    img[:, :, 1] += MEAN[1]
    img[:, :, 2] += MEAN[2]
    img = img.astype(np.uint8)
    return img


def ImageList2np_img_list(ImageList):
    """Convert ImageList to np_img list.

    :param ImageList:
    :return: [np_img1, np_img2, ... ]
    """
    img_list = []

    # ImageList.tensors [N, 3, H, W]
    for tensor in ImageList.tensors:
        # tensor [3, H, W]
        img_list.append(img_tensor2np_img(tensor))

    return img_list


def cv_show(im, name="cv_show", pause=1, W=500, H=500):
    """show img with opencv.

    :param im: [3, H, W]
    :param name: title
    :return:
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, W, H)
    cv2.imshow(name, im)
    cv2.waitKey(pause)


def mem2feature(m, shape=("H", "W")):
    """reshape saved memory back to feature shape.

    :param m: tensor [l, C]
    :param shape: (H, W) feature map shape, H*W = l
    :return: tensor [C, H, W]
    """
    m = m.reshape(shape[0], shape[1], -1).permute(2, 0, 1).detach().cpu()
    return m


def plt_show(im, name=None, norm=True):
    """
    :param im:
    :param name:
    :param norm: whether scale im into [0, 1]
    :return:
    """
    if name:
        plt.figure(name)
    else:
        plt.figure()
    if norm:
        plt.imshow(im)
    else:
        plt.imshow(im, vmin=0, vmax=1)
    plt.show()
    return


def feature2im(tensor, norm=True):
    """covert pytroch feature map to ndarray.

    :param tensor: [C, H, W]
    :param norm: bool whether normalize result to [0,1]
    :return: [H, W] gray img
    """
    _t = tensor.clone()
    _t = _t.norm(2, dim=0)
    im = tensor2np(_t)
    if norm:
        im /= im.max()
    else:
        im /= max(im.max(), 1)
    return im


def filter_conns(ndarray):
    num_layers, num_features, num_conns = ndarray.shape
    levels_perserve = [num_features - 1]
    for i in range(num_layers, 0, -1):
        for level in range(num_features):
            if level not in levels_perserve:
                ndarray[i][level] = 0
        for level in levels_perserve:
            ndarray[i][level] = np.where(
                ndarray[i, level] == ndarray[i, level].max(), 1, 0
            )

    return ndarray


def norm_conns(ndarray):
    num_layers, num_features, num_conns = ndarray.shape
    for i in range(num_layers):
        for level in range(num_features):
            vmin = ndarray[i, level].min()
            vmax = ndarray[i, level].max()
            ndarray[i][level] = (ndarray[i][level] - vmin) / (vmax - vmin)
    return ndarray


def conns_to_edge(i, j, k, num_features):
    """from position of conns tensor (i, j, k) to edge (node_in, node_out)"""
    # num_layers, num_features, num_conns = ndarray.shape
    node_level = j
    node_out = (i + 1) * num_features + node_level

    if k == 0:
        node_in_level = node_level
    elif k <= node_level:
        node_in_level = k - 1
    else:
        node_in_level = k

    node_in = i * num_features + node_in_level

    return node_in, node_out, node_in_level
