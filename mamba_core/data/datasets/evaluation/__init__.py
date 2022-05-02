from mamba_core.data import datasets

from .cityscapes import abs_cityscapes_evaluation
from .coco import coco_evaluation
from .vid import vid_evaluation
from .voc import voc_evaluation


def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.AbstractDataset):
        return abs_cityscapes_evaluation(**args)
    elif isinstance(dataset, datasets.VIDDataset):
        return vid_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError(f"Unsupported dataset type {dataset_name}.")
