# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import numpy as np
import os
import torch
from mamba_core.data.datasets.evaluation import evaluate
from tqdm import tqdm

from ..utils.comm import (
    all_gather,
    get_rank,
    get_world_size,
    is_main_process,
    synchronize,
)
from ..utils.debugger import Debugger
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug


def debug(
    images, gt_boxes, output, iter_id, cfg=None, filename=None, output_folder=None
):
    MEAN = [102.9801, 115.9465, 122.7717]
    # get first image and output
    # postprocess to get boxes of first image
    if isinstance(images, dict):
        batch = images["cur"].tensors[0]
    else:
        batch = images.tensors[0]
    dets = output[0].bbox
    dets = dets.detach().cpu().numpy().reshape(1, -1, 4)
    if output[0].has_field("scores"):
        scores = output[0].get_field("scores").detach().cpu().numpy()
    else:
        scores = output[0].get_field("objectness").detach().cpu().numpy()

    if output[0].has_field("labels"):
        labels = output[0].get_field("labels").detach().cpu().numpy()
    else:
        # rpn only no labels
        labels = np.ones_like(scores)

    gt_det = gt_boxes[0].bbox
    gt_det = gt_det.detach().cpu().numpy().reshape(1, -1, 4)
    dets_gt = (
        np.array(gt_det, dtype=np.float32)
        if len(gt_det) > 0
        else np.zeros((1, 6), dtype=np.float32)
    )

    labels_gt = gt_boxes[0].get_field("labels").detach().cpu().numpy()

    for i in range(1):
        debugger = Debugger(dataset="VID")
        img = batch.detach().cpu().numpy().transpose(1, 2, 0)
        img[:, :, 0] += MEAN[0]
        img[:, :, 1] += MEAN[1]
        img[:, :, 2] += MEAN[2]
        img = img.astype(np.uint8)
        debugger.add_img(img, img_id="out_pred")
        for k in range(len(dets[i])):
            if scores[k] > 0.3:
                debugger.add_coco_bbox(
                    dets[i, k, :4], labels[k], scores[k], img_id="out_pred"
                )

        debugger.add_img(img, img_id="out_gt")
        for k in range(len(dets_gt[i])):
            debugger.add_coco_bbox(
                dets_gt[i, k, :4], labels_gt[k], 1.0, img_id="out_gt"
            )

        if cfg.TEST.DEBUG.LEVEL == 1:
            # save all imgs
            debugger.save_all_imgs(
                os.path.join(output_folder, "visualization"), prefix=filename
            )
        elif cfg.TEST.DEBUG.LEVEL == 2:
            # show with opencv
            debugger.show_all_imgs(pause=cfg.TEST.DEBUG.PAUSE, step=iter_id)
        # elif cfg.DEBUG.debug == 5:
        #     # show with tensorboard
        #     debugger.show_all_imgs(logger=logger, step=iter_id, down_scale=True)
        else:
            raise ValueError("Unexpected cfg.TEST.DEBUG.LEVEL value.")


def compute_on_dataset(
    model,
    data_loader,
    device,
    bbox_aug,
    method,
    timer=None,
    cfg=None,
    output_folder=None,
):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                if method in ("base",):
                    images = images.to(device)
                elif method in ("rdn", "mega", "fgfa", "dff", "mamba", "selsa"):
                    images["cur"] = images["cur"].to(device)
                    for key in ("ref", "ref_l", "ref_m", "ref_g"):
                        if key in images.keys():
                            images[key] = [img.to(device) for img in images[key]]
                else:
                    raise ValueError(f"method {method} not supported yet.")
                output = model(images)

            if cfg.TEST.DEBUG.LEVEL > 0 and get_rank() == 0:
                try:
                    filename = images["filename"] + ".PNG"
                except RuntimeError:
                    filename = data_loader.dataset.image_set_index[i] + ".PNG"
                debug(images, targets, output, i, cfg, filename, output_folder)

            if timer:
                if not device.type == "cpu":
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("mamba_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
    cfg,
    model,
    data_loader,
    dataset_name,
    iou_types=("bbox",),
    motion_specific=False,
    box_only=False,
    bbox_aug=False,
    device="cuda",
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("mamba_core.inference")
    dataset = data_loader.dataset
    logger.info(
        "Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset))
    )
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(
        model,
        data_loader,
        device,
        bbox_aug,
        cfg.MODEL.VID.METHOD,
        inference_timer,
        cfg=cfg,
        output_folder=output_folder,
    )
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        motion_specific=motion_specific,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        **extra_args,
    )


def inference_no_model(
    data_loader,
    iou_types=("bbox",),
    motion_specific=False,
    box_only=False,
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
):
    dataset = data_loader.dataset

    predictions = torch.load(os.path.join(output_folder, "predictions.pth"))
    print("prediction loaded.")

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        motion_specific=motion_specific,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        **extra_args,
    )
