import cv2
import numpy as np
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

classes = [
    "__background__",  # always index 0
    "airplane",
    "antelope",
    "bear",
    "bicycle",
    "bird",
    "bus",
    "car",
    "cattle",
    "dog",
    "domestic_cat",
    "elephant",
    "fox",
    "giant_panda",
    "hamster",
    "horse",
    "lion",
    "lizard",
    "monkey",
    "motorcycle",
    "rabbit",
    "red_panda",
    "sheep",
    "snake",
    "squirrel",
    "tiger",
    "train",
    "turtle",
    "watercraft",
    "whale",
    "zebra",
]
classes_map = [
    "__background__",  # always index 0
    "n02691156",
    "n02419796",
    "n02131653",
    "n02834778",
    "n01503061",
    "n02924116",
    "n02958343",
    "n02402425",
    "n02084071",
    "n02121808",
    "n02503517",
    "n02118333",
    "n02510455",
    "n02342885",
    "n02374451",
    "n02129165",
    "n01674464",
    "n02484322",
    "n03790512",
    "n02324045",
    "n02509815",
    "n02411705",
    "n01726692",
    "n02355227",
    "n02129604",
    "n04468005",
    "n01662784",
    "n04530566",
    "n02062744",
    "n02391049",
]

num_classes = len(classes)

classes_to_ind = dict(zip(classes_map, range(len(classes_map))))


def get_annotation_of_img(image_path):
    roi_rec = dict()
    filename = image_path.replace("JPEG", "xml")
    filename = filename.replace("Data", "Annotations")
    tree = ET.parse(filename)
    size = tree.find("size")
    roi_rec["height"] = float(size.find("height").text)
    roi_rec["width"] = float(size.find("width").text)

    objs = tree.findall("object")
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.float32)
    valid_objs = np.zeros((num_objs), dtype=np.bool)

    class_to_index = dict(zip(classes_map, range(num_classes)))
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find("bndbox")
        # Make pixel indexes 0-based
        x1 = np.maximum(float(bbox.find("xmin").text), 0)
        y1 = np.maximum(float(bbox.find("ymin").text), 0)
        x2 = np.minimum(float(bbox.find("xmax").text), roi_rec["width"] - 1)
        y2 = np.minimum(float(bbox.find("ymax").text), roi_rec["height"] - 1)
        if obj.find("name").text not in class_to_index.keys():
            continue
        valid_objs[ix] = True
        boxes[ix, :] = [x1, y1, x2, y2]

    boxes = boxes[valid_objs, :]
    return boxes


# def get_annotation_of_img(filename):
#     filename = filename.replace('JPEG', 'xml')
#     filename = filename.replace('Data', 'Annotations')
#     tree = ET.parse(filename)
#     boxes = []
#     gt_classes = []
#     size = tree.find('size')
#     im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
#
#     objs = tree.findall("object")
#     for obj in objs:
#         if not obj.find("name").text in classes_to_ind:
#             continue
#
#         bbox = obj.find("bndbox")
#         box = [
#             np.maximum(float(bbox.find("xmin").text), 0),
#             np.maximum(float(bbox.find("ymin").text), 0),
#             np.minimum(float(bbox.find("xmax").text), im_info[1] - 1),
#             np.minimum(float(bbox.find("ymax").text), im_info[0] - 1)
#         ]
#         boxes.append(box)
#         gt_classes.append(classes_to_ind[obj.find("name").text.lower().strip()])
#
#     res = {
#         "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
#         "labels": torch.tensor(gt_classes),
#         "im_info": im_info,
#     }
#     return res


def resize(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """only resize input image to target size and return scale.

    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(
        im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation
    )

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        return padded_im, im_scale


def resize_to(im, target_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """only resize input image to target size and return scale.

    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape[:2]
    im_scale = np.zeros((2,), dtype=np.float32)
    im_scale[1] = float(target_size[0]) / float(im_shape[0])
    im_scale[0] = float(target_size[1]) / float(im_shape[1])
    # prevent bigger axis from being more than max_size:
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, (target_size[1], target_size[0]), interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        return padded_im, im_scale
