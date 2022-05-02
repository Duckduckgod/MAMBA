# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import torch
import torchvision
from mamba_core.structures.bounding_box import BoxList
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class Resize:
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image, target
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
        self.chance = 0.0

    def __call__(self, image, target=None):
        if target is not None:
            self.chance = random.random()
        if self.chance < self.prob:
            image = F.hflip(image)
            if target is not None:
                target = target.transpose(0)

        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.vflip(image)
            if target is not None:
                target = target.transpose(1)
        return image, target


class ColorJitter:
    def __init__(
        self,
        brightness=None,
        contrast=None,
        saturation=None,
        hue=None,
    ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, target=None):
        image = self.color_jitter(image)
        return image, target


class ToTensor:
    def __call__(self, image, target=None):
        return F.to_tensor(image), target


class Normalize:
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, target
        return image, target


class LightningNoise:
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target=None):
        if random.random() < 0.5:
            swap = self.perms[random.randint(0, len(self.perms) - 1)]
            new_img = F.to_tensor(image)
            new_img = new_img[swap, :, :]
            image = F.to_pil_image(new_img)

        return image, target


# ==========================BEGIN CACULATE IoU==================================
def intersect(boxes1, boxes2):
    """
    Find intersection of every box combination between two sets of box
    boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
    boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)

    Out: Intersection each of boxes1 with respect to each of boxes2,
         a tensor of dimensions (n1, n2)
    """
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy = torch.min(
        boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
        boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2),
    )

    min_xy = torch.max(
        boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
        boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2),
    )
    inter = torch.clamp(max_xy - min_xy, min=0)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  # (n1, n2)


def find_IoU(boxes1, boxes2):
    """
    Find IoU between every boxes set of boxes
    boxes1: a tensor of dimensions (n1, 4) (left, top, right , bottom)
    boxes2: a tensor of dimensions (n2, 4)

    Out: IoU each of boxes1 with respect to each of boxes2, a tensor of
         dimensions (n1, n2)

    Formula:
    (box1 ∩ box2) / (box1 u box2) = (box1 ∩ box2) / (area(box1) + area(box2) - (box1 ∩ box2 ))
    """
    inter = intersect(boxes1, boxes2)
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    area_boxes1 = area_boxes1.unsqueeze(1).expand_as(inter)  # (n1, n2)
    area_boxes2 = area_boxes2.unsqueeze(0).expand_as(inter)  # (n1, n2)
    union = area_boxes1 + area_boxes2 - inter
    return inter / union


# ==========================END CACULATE IoU====================================


def random_crop(image, boxes, labels):
    """
    Performs a random crop. Helps to learn to detect larger and partial object
    image: A tensor of dimensions (3, original_h, original_w)
    boxes: Bounding boxes, a tensor of dimensions (n_objects, 4)
    labels: labels of object, a tensor of dimensions (n_objects)
    difficulties: difficulties of detect object, a tensor of dimensions (n_objects)

    Out: cropped image (Tensor), new boxes, new labels, new difficulties
    """
    # if type(image) == PIL.Image.Image:
    #     image = F.to_tensor(image)
    image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    aspect_ratio = float(original_h) / float(original_w)

    while True:
        mode = random.choice([0.1, 0.3, 0.5, 0.9, None])

        if mode is None:
            return image, boxes, labels

        # new_image = image
        # new_boxes = boxes
        # new_difficulties = difficulties
        # new_labels = labels
        for _ in range(50):
            # Crop dimensions: [0.3, 1] of original dimensions
            # new_h = random.uniform(0.3 * original_h, original_h)
            new_w = random.uniform(0.3 * original_w, original_w)
            new_h = new_w * aspect_ratio

            # # Aspect ratio constraint b/t .5 & 2
            # if new_h / new_w < 0.5 or new_h / new_w > 2:
            #     continue

            # Crop coordinate
            left = random.uniform(0, original_w - new_w)
            right = left + new_w
            top = random.uniform(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])

            # Calculate IoU  between the crop and the bounding boxes
            overlap = find_IoU(crop.unsqueeze(0), boxes)  # (1, n_objects)
            overlap = overlap.squeeze(0)
            # If not a single bounding box has a IoU of greater than the minimum, try again
            if overlap.max().item() < mode:
                continue

            # Crop
            new_image = image[
                :, int(top) : int(bottom), int(left) : int(right)
            ]  # (3, new_h, new_w)

            # Center of bounding boxes
            center_bb = (boxes[:, :2] + boxes[:, 2:]) / 2.0

            # Find bounding box has been had center in crop
            center_in_crop = (
                (center_bb[:, 0] > left)
                * (center_bb[:, 0] < right)
                * (center_bb[:, 1] > top)
                * (center_bb[:, 1] < bottom)
            )  # (n_objects)

            if not center_in_crop.any():
                continue

            # take matching bounding box
            new_boxes = boxes[center_in_crop, :]

            # take matching labels
            new_labels = labels[center_in_crop]

            # # take matching difficulities
            # new_difficulties = difficulties[center_in_crop]

            # Use the box left and top corner or the crop's
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])

            # adjust to crop
            new_boxes[:, :2] -= crop[:2]

            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])

            # adjust to crop
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels

        # return new_image, new_boxes, new_labels


def random_crop_ref(image):
    # if type(image) == PIL.Image.Image:
    #     image = F.to_tensor(image)
    image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    aspect_ratio = float(original_h) / float(original_w)

    while True:
        mode = random.choice([0.1, 0.3, 0.5, 0.9, None])

        if mode is None:
            return image

        # new_image = image
        # new_difficulties = difficulties
        for _ in range(50):
            # Crop dimensions: [0.3, 1] of original dimensions
            # new_h = random.uniform(0.3 * original_h, original_h)
            new_w = random.uniform(0.3 * original_w, original_w)
            new_h = new_w * aspect_ratio

            # # Aspect ratio constraint b/t .5 & 2
            # if new_h / new_w < 0.5 or new_h / new_w > 2:
            #     continue

            # Crop coordinate
            left = random.uniform(0, original_w - new_w)
            right = left + new_w
            top = random.uniform(0, original_h - new_h)
            bottom = top + new_h

            # Crop
            new_image = image[
                :, int(top) : int(bottom), int(left) : int(right)
            ]  # (3, new_h, new_w)

            return new_image

        # return new_image


class RandCrop:
    def __call__(self, image, target=None):
        if target is None:
            image = random_crop_ref(image)
            image = F.to_pil_image(image)
            new_target = None
        else:
            boxes = target.bbox
            labels = target.get_field("labels")
            image, boxes, labels = random_crop(image, boxes, labels)

            image = F.to_pil_image(image)
            new_target = BoxList(boxes, image.size, target.mode)
            new_target.add_field("labels", labels)

        return image, new_target


# Expand with filler
def expand_filler(image, boxes, filler, max_scale):
    """
    Perform a zooming out operation by placing the
    image in a larger canvas of filler material. Helps to learn to detect
    smaller objects.
    image: input image, a tensor of dimensions (3, original_h, original_w)
    boxes: bounding boxes, a tensor of dimensions (n_objects, 4)
    filler: RBG values of the filler material, a list like [R, G, B]

    Out: new_image (A Tensor), new_boxes
    """
    # if type(image) == PIL.Image.Image:
    #     image = F.to_tensor(image)
    image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(
        1
    ).unsqueeze(1)

    # Place the original image at random coordinates
    # in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h

    new_image[:, top:bottom, left:right] = image

    # Adjust bounding box
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_image, new_boxes


def expand_filler_ref(image, filler, max_scale):
    # if type(image) == PIL.Image.Image:
    #     image = F.to_tensor(image)
    image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(
        1
    ).unsqueeze(1)

    # Place the original image at random coordinates
    # in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h

    new_image[:, top:bottom, left:right] = image

    return new_image


class ExpandFiller:
    def __init__(self, max_scale=2.0):
        self.mean = [102.9801 / 255, 115.9465 / 255, 122.7717 / 255]
        self.max_scale = max_scale

    def __call__(self, image, target=None):
        if random.random() < 0.5:
            return image, target

        if target is None:
            image = expand_filler_ref(image, self.mean, self.max_scale)
            image = F.to_pil_image(image)
            new_target = None
        else:
            boxes = target.bbox
            image, boxes = expand_filler(image, boxes, self.mean, self.max_scale)
            image = F.to_pil_image(image)

            labels = target.get_field("labels")
            new_target = BoxList(boxes, image.size, target.mode)
            new_target.add_field("labels", labels)

        return image, new_target
