import json
import os
from collections import defaultdict

import cv2
import numpy as np
from megengine.data.dataset import VisionDataset


class CrowdHuman(VisionDataset):
    supported_order = (
        "image",
        "boxes",
        "vboxes",
        "hboxes",
        "boxes_category",
        "info",
    )

    def __init__(self, root, ann_file, remove_images_without_annotations=True, *, order=None):
        super().__init__(root, order=order, supported_order=self.supported_order)
        print('load annotation file: ', ann_file)
        with open(ann_file, "r") as f:
            dataset = json.load(f)

        self.imgs = dict()
        for img in dataset["images"]:
            self.imgs[img["id"]] = img

        self.imgs_with_anns = defaultdict(list)
        for ann in dataset["annotations"]:
            self.imgs_with_anns[ann["image_id"]].append(ann)

        self.cats = dict()
        for cat in dataset["categories"]:
            self.cats[cat["id"]] = cat

        self.ids = list(sorted(self.imgs.keys()))  # A list contains keys

        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = self.imgs_with_anns[img_id]
                if len(anno) == 0:
                    del self.imgs[img_id]
                    del self.imgs_with_anns[img_id]
                else:
                    ids.append(img_id)

            self.ids = ids
        print("load with order", self.order)

    def __getitem__(self, index):
        img_id = self.ids[index]
        anno = self.imgs_with_anns[img_id]

        target = []
        for k in self.order:
            if k == "image":
                file_name = self.imgs[img_id]["file_name"]
                path = os.path.join(self.root, file_name)
                image = cv2.imread(path, cv2.IMREAD_COLOR)  # BRG
                target.append(image)
            elif k == "boxes":
                boxes = [obj["bbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                # transfer boxes from xywh to xyxy
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "vboxes":
                boxes = [obj["vbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "hboxes":
                boxes = [obj["hbox"] for obj in anno]
                boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
                boxes[:, 2:] += boxes[:, :2]
                target.append(boxes)
            elif k == "boxes_category":
                boxes_category = [obj["category_id"] for obj in anno]
                boxes_category = np.array(boxes_category, dtype=np.int32)
                target.append(boxes_category)
            elif k == "info":
                info = self.imgs[img_id]
                info = [info["height"], info["width"], info["file_name"]]
                target.append(info)

        return tuple(target)

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.ids[index]
        img_info = self.imgs[img_id]
        return img_info

    class_names = ("preson",
                   "mask")

