import argparse
import json
import os

import cv2 as cv


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", '--data-path',
        default=None,
        type=str,
        help='the path of CrowdHuman dataset'
    )
    parser.add_argument(
        "-o", "--odgt-path",
        default='xxx.odgt',
        type=str,
        help="the path of CrowdHuman odgt file"
    )
    parser.add_argument(
        "-s", "--save-path",
        default='xxx.json',
        type=str,
        help='the path to save json file'
    )
    parser.add_argument(
        "-v", "--visible",
        default=0,
        type=int,
        help="keep visible box",
    )
    parser.add_argument(
        "-f", "--full",
        default=1,
        type=int,
        help="keep full box",
    )
    parser.add_argument(
        "--head",
        default=0,
        type=int,
        help="keep head box",
    )
    parser.add_argument(
        "--rm-occ",
        default=1,
        type=int,
        help="remove occluded box",
    )
    parser.add_argument(
        "--rm-hignore",
        default=0,
        type=int,
        help="remove ignored head box",
    )
    parser.add_argument(
        "--rm-hocc",
        default=0,
        type=int,
        help="remove occluded head head",
    )
    parser.add_argument(
        "--rm-hunsure",
        default=0,
        type=int,
        help="keep unsure head box",
    )
    parser.add_argument(
        "--tag",
        default="person",
        type=str,
        help="keep box with tag 'person' or 'mask' or 'both'"
    )
    return parser


def readlines(filename):
    print("start read odgt file ")
    with open(filename, 'r') as f:
        lines = f.readlines()
    name = filename.split(os.sep)[-1].split('.')[0].split('_')[-1]
    print(f"{len(lines)} images in CrowdHuman {name} dataset")

    return [json.loads(line.strip('\n')) for line in lines]


def crowdhuman2coco(args, odgt_path, json_path, data_path):
    records = readlines(odgt_path)  # A list contains dicts
    json_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }  # coco format
    bbox_id = 1
    categories = {}
    print("start convert")
    for image_id, image_dict in enumerate(records):
        file_name = image_dict['ID'] + '.jpg'
        im = cv.imread(data_path + file_name, 0)  # gain height and width
        image = {
            'file_name': file_name,
            'height': im.shape[0],
            'width': im.shape[1],
            'id': image_dict['ID']
        }
        json_dict['images'].append(image)
        gt_box = image_dict['gtboxes']  # A list contains dicts

        for _, instance in enumerate(gt_box):
            annotation = {}
            category = instance['tag']

            if category not in categories:
                new_id = len(categories) + 1
                categories[category] = new_id

            if instance['tag'] == args.tag or 'both' == args.tag:
                annotation['category_id'] = categories[category]
            else:
                continue

            if args.full:
                attr = instance['extra']
                if args.rm_occ and attr['ignore']:
                    continue
                annotation['bbox'] = instance['fbox']
            if args.visible:
                attr = instance['extra']
                if args.rm_occ and attr['ignore']:
                    continue
                annotation['vbox'] = instance['vbox']
            if args.head:
                attr = instance['head_attr']
                if args.rm_hocc and attr['occ']:
                    continue
                if args.rm_hunsure and attr['unsure']:
                    continue
                if args.rm_hignore and attr['ignore']:
                    continue
                annotation['hbox'] = instance['hbox']

            annotation['image_id'] = image_dict['ID']
            annotation['id'] = bbox_id
            bbox_id += 1
            json_dict['annotations'].append(annotation)

    for cate, cid in categories.items():
        cat = {
            'supercategory': cate,
            'id': cid,
            'name': cate
        }
        json_dict['categories'].append(cat)

    print("start write json")
    json_fp = open(json_path, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print(f"Json file have been dumped to {json_path}")


def main():
    parser = make_parser()
    args = parser.parse_args()
    crowdhuman2coco(args, args.odgt_path, args.save_path, args.data_path)


if __name__ == "__main__":
    main()
