import numpy as np
from PIL import Image

from mamba_core.config import cfg

from .vid import VIDDataset


class VIDSELSADataset(VIDDataset):
    def __init__(
        self,
        image_set,
        data_dir,
        img_dir,
        anno_path,
        img_index,
        transforms,
        is_train=True,
    ):
        super().__init__(
            image_set,
            data_dir,
            img_dir,
            anno_path,
            img_index,
            transforms,
            is_train=is_train,
        )
        self.load_anno_for_ref = cfg.INPUT.USE_SELSA_CODE
        if not self.is_train:
            self.start_index = []
            self.start_id = []
            # shuffled index
            self.shuffled_index = {}
            for id, image_index in enumerate(self.image_set_index):
                frame_id = int(image_index.split("/")[-1])
                if frame_id == 0:
                    self.start_index.append(id)

                    # save shuffled frames
                    shuffled_index = np.arange(self.frame_seg_len[id])
                    np.random.shuffle(shuffled_index)
                    self.shuffled_index[str(id)] = shuffled_index

                    self.start_id.append(id)
                else:
                    self.start_id.append(self.start_index[-1])

    def _get_train(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        # if a video dataset
        img_refs_g = []
        if self.load_anno_for_ref:
            ref_filenames = []
        if hasattr(self, "pattern"):
            ref_ids = np.random.choice(
                self.frame_seg_len[idx], cfg.MODEL.VID.MAMBA.REF_NUM, replace=False
            )
            for ref_id in ref_ids:
                ref_filename = self.pattern[idx] % ref_id
                img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                img_refs_g.append(img_ref)
                if self.load_anno_for_ref:
                    ref_filenames.append(self._img_dir % ref_filename)
        else:
            # det dataset
            for i in range(cfg.MODEL.VID.MAMBA.REF_NUM):
                img_refs_g.append(img.copy())
                if self.load_anno_for_ref:
                    ref_filenames.append(self._img_dir % filename)

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_g)):
                if self.load_anno_for_ref:
                    img_with_name = {"img": img_refs_g[i], "filename": ref_filenames[i]}
                    img_refs_g[i], _ = self.transforms(img_with_name, None)
                else:
                    img_refs_g[i], _ = self.transforms(img_refs_g[i], None)

        images = {}
        images["cur"] = img
        images["ref"] = img_refs_g

        return images, target, idx

    def _get_test(self, idx):
        filename = self.image_set_index[idx]
        # give the current frame a category. 0 for start, 1 for normal
        frame_id = int(filename.split("/")[-1])
        frame_category = 1
        if frame_id == 0:
            frame_category = 0
        # change filename with shuffled sequence
        shuffled_index = self.shuffled_index[str(self.start_id[idx])]
        frame_seg_id = self.frame_seg_id[idx]
        shuffled_frame_id = int(shuffled_index[frame_seg_id])
        filename = self.pattern[idx] % shuffled_frame_id
        idx = self.start_id[idx] + shuffled_frame_id
        img = Image.open(self._img_dir % filename).convert("RGB")

        img_refs = []
        # reading other images of the queue (not necessary to be the last one, but last one here)
        ref_seg_id = min(
            self.frame_seg_len[idx] - 1, frame_seg_id + cfg.MODEL.VID.RDN.MAX_OFFSET
        )
        shuffled_ref_frame_id = int(shuffled_index[ref_seg_id])
        ref_filename = self.pattern[idx] % shuffled_ref_frame_id
        img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
        img_refs.append(img_ref)

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs)):
                img_refs[i], _ = self.transforms(img_refs[i], None)

        images = {}
        images["cur"] = img
        images["ref"] = img_refs
        images["frame_category"] = frame_category
        images["seg_len"] = self.frame_seg_len[idx]
        images["shuffled_index"] = shuffled_index
        images["pattern"] = self.pattern[idx]
        images["img_dir"] = self._img_dir
        images["transforms"] = self.transforms
        images["filename"] = filename

        return images, target, idx
