# CrowdHuman

This repo contains a script to convert the CrowdHuman dataset annotations to COCO format and a `dataset Class` for reading data.

## Introduction

[CrowdHuman](https://www.crowdhuman.org/) is a benchmark dataset to better evaluate detectors in crowd scenarios. The CrowdHuman dataset is large, rich-annotated and contains high diversity. CrowdHuman contains 15000, 4370 and 5000 images for training, validation, and testing, respectively. There are a total of 470K human instances from train and validation subsets and 23 persons per image, with various kinds of occlusions in the dataset. Each human instance is annotated with a head bounding-box, human visible-region bounding-box and human full-body bounding-box. We hope our dataset will serve as a solid baseline and help promote future research in human detection tasks.

## Dataset

### Annotation format

Supported `annotation_train.odgt` and `annotation_val.odgt` which contains the annotations of CrowdHuman.

`odgt` is a file format that **each line of it is a JSON**, this JSON contains the whole annotations for the relative image. We prefer using this format since it is reader-friendly.

**Annotation format:**

```python
JSON{
    "ID" : image_filename,
    "gtboxes" : [gtbox], 
}

gtbox{
    "tag" : "person" or "mask", 
    "vbox": [x, y, w, h],
    "fbox": [x, y, w, h],
    "hbox": [x, y, w, h],
    "extra" : extra, 
    "head_attr" : head_attr, 
}

extra{
    "ignore": 0 or 1,
    "box_id": int,
    "occ": int,
}

head_attr{
    "ignore": 0 or 1,
    "unsure": int,
    "occ": int,
}
```

- `Keys` in `extra` and `head_attr` are **optional**, it means some of them may not exist
- `extra/head_attr` contains attributes for `person/head`
- `tag` is `mask` means that this box is `crowd/reflection/something like person/...` and need to be `ignore`(the `ignore` in `extra` is `1`)
- `vbox, fbox, hbox` means `visible box, full box, head box` respectively

### Download

[CrowdHuman_train01.zip](https://drive.google.com/file/d/134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y/view)

[CrowdHuman_train02.zip](https://drive.google.com/file/d/17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla/view)

[CrowdHuman_train03.zip](https://drive.google.com/file/d/1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW/view)

[CrowdHuman_val.zip](https://drive.google.com/file/d/18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO/view)

[annotation_train.odgt](https://drive.google.com/file/d/1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3/view)

[annotation_val.odgt](https://drive.google.com/file/d/10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL/view)

[CrowdHuman_test.zip](https://drive.google.com/file/d/1tQG3E_RrRI4wIGskorLTmDiWHH2okVvk/view)

## Get Started

### Convert annotations

Before converting, ensure that the dataset folder format like this:

```sh
|- crowdhuman
 |- Images #contain all train and test images
 |- annotation_train.json
 |- annotation_val.json
```

You can use this to simply keep full boxes with tag `person` only

```shell
python crowdhuman2coco.py -d /path/to/crowdhuman/dataset -o /path/to/annotation_train.odgt/ -j /path/to/annotation_train.json
```

For more demand, run this to get more detial infomation

```shell
python crowdhuman2coco.py --help
```

### Simple Dataset

This repo also contains two simple implement of `CrowdHuman Dataset Class` in PyTorch and MegEngine.

The Dataset will return a tuple that contains the annotations that you need  in order everytime when it call `__getitem__`

**supported_order**

```python
class CrowdHuman(VisionDataset):
    supported_order = (
        "image",
        "boxes",
        "vboxes",
        "hboxes",
        "boxes_category",
        "info",
    )
```

You can easily use this to instantiate a crowdhuman_dataset

```python
crowdhuman_dataset = CrowdHuman(
    root='path/to/CrowdHuman',
    ann_file='path/to/annotations.json',
    remove_images_without_annotations=True,
    order=[
        'image',
        'boxes',
        'boxes_category'
        'info'
    ]
)
```

