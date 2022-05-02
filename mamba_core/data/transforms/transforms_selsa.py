# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import numpy as np
import torch
import types
from mamba_core.structures.bounding_box import BoxList
from numpy import random
from PIL import Image

from .image_utils import get_annotation_of_img, resize_to


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda:
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts:
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class ConvertToInts:
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.int), boxes, labels


class SubtractMeans:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords:
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords:
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize:
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue:
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise:
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor:
    def __init__(self, current="BGR", transform="HSV"):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == "BGR" and self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == "HSV" and self.transform == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness:
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image:
    def __call__(self, tensor, boxes=None, labels=None):
        return (
            tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)),
            boxes,
            labels,
        )


class ToTensor:
    def __call__(self, cvimage, boxes=None, labels=None):
        return (
            torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1),
            boxes,
            labels,
        )


class RandomSampleCrop:
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self, crop_pert=0.3, no_iou_limit=False):
        self.crop_pert = crop_pert
        self.no_iou_limit = no_iou_limit
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        aspect_ratio = float(height) / float(width)
        while True:
            # randomly choose a mode
            mode = random.choice(np.asarray(self.sample_options, dtype="object"))
            if self.no_iou_limit:
                mode = (None, None)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float("-inf")
            if max_iou is None:
                max_iou = float("inf")

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(self.crop_pert * width, width)
                h = w * aspect_ratio

                # # aspect ratio constraint b/t .5 & 2
                # if h / w < 0.5 or h / w > 2:
                #     continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1] : rect[3], rect[0] : rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                # current_labels = labels[mask]
                current_labels = np.zeros_like(labels, dtype=bool)
                current_labels[mask] = True

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand:
    def __init__(self, mean, expand_scale=2.0):
        self.mean = mean
        self.expand_scale = expand_scale

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        float(height) / float(width)
        ratio = random.uniform(1, self.expand_scale)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth), dtype=image.dtype
        )
        expand_image[:, :, :] = self.mean
        expand_image[
            int(top) : int(top + height), int(left) : int(left + width)
        ] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror:
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels:
    """Transforms a tensorized image by swapping the channels in the order
    specified in the swap tuple.

    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort:
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform="HSV"),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current="HSV", transform="BGR"),
            RandomContrast(),
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation:
    def __init__(
        self,
        size=300,
        mean=(104, 117, 123),
        expand_scale=2,
        crop_pert=0.3,
        color=False,
        no_iou_limit=False,
    ):
        self.mean = mean
        self.size = size
        self.expand_scale = expand_scale
        self.crop_pert = crop_pert
        self.color = color
        self.no_iou_limit = no_iou_limit
        if self.color:
            self.augment = Compose(
                [
                    ConvertFromInts(),
                    PhotometricDistort(),
                    Expand(self.mean, self.expand_scale),
                    RandomSampleCrop(self.crop_pert, self.no_iou_limit),
                ]
            )
        else:
            self.augment = Compose(
                [
                    ConvertFromInts(),
                    Expand(self.mean, self.expand_scale),
                    RandomSampleCrop(self.crop_pert, self.no_iou_limit),
                ]
            )

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: cv2 image, BGR, dtype np.UINT8
            boxes: boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            labels: label = np.ones((num_objs, ), dtype=bool)
        Returns:
            img, boxes, labels
        """
        return self.augment(img, boxes, labels)


class SSDAugWrapper:
    def __init__(
        self,
        mean=(104, 117, 123),
        expand_scale=2,
        crop_pert=0.3,
        color=True,
        no_iou_limit=False,
    ):
        self.SSDAugmentor = SSDAugmentation(
            mean=mean,
            expand_scale=expand_scale,
            crop_pert=crop_pert,
            color=color,
            no_iou_limit=no_iou_limit,
        )

    def __call__(self, pil_img, target):
        """
        Args:
            pil_img: PIL image RGB
            target: None or boxlist
        Returns:
            pil_img, target
        """
        # transforms for ref frame
        if target is None:
            filename = pil_img["filename"]
            pil_img = pil_img["img"]
            boxes = get_annotation_of_img(filename)
        else:
            # get boxes
            boxes = target.bbox.numpy()
            # get boxes
            ori_labels = target.get_field("labels")

        # covert to array
        img = np.array(pil_img)
        # RGB to BGR
        img = img[:, :, ::-1]
        # get size
        ori_size = (img.shape[0], img.shape[1])

        # has an object
        if boxes.shape[0] != 0:
            label = np.ones((boxes.shape[0],), dtype=bool)
            im_aug, bbs_aug, mask = self.SSDAugmentor(img.copy(), boxes.copy(), label)
            im_aug = im_aug.astype(np.uint8)
            im_aug, im_scale = resize_to(im_aug, ori_size)
            bbs_aug[:, :4] *= [im_scale[0], im_scale[1], im_scale[0], im_scale[1]]
        # if no object
        else:
            bbs_aug = None
            im_aug = img

        # prepare target for key frame
        if target is not None:
            # if has selected boxes
            if np.sum(mask) != 0:
                assert bbs_aug.shape[0] != 0
                # prepare new target
                new_labels = ori_labels[mask]
                target = BoxList(bbs_aug.reshape(-1, 4), ori_size[::-1], mode="xyxy")
                target.add_field("labels", new_labels)
            else:
                im_aug = img

        # BGR to RGB
        img = im_aug[:, :, ::-1]
        # convert to PIL image
        img = Image.fromarray(img)

        return img, target
