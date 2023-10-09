"""# Coco Merge App.

The [`pyodi coco`][pyodi.apps.coco.coco_merge.coco_merge] app can be used to merge COCO annotation files.

Example usage:

``` bash
pyodi coco merge coco_1.json coco_2.json output.json
```

This app merges COCO annotation files by replacing original image and annotations ids with new ones
and adding all existent categories.

---

# API REFERENCE
"""  # noqa: E501
import json
from typing import Any, Dict, Optional

from loguru import logger
"""
1. saves images/annotations from categories
2. creates new json by filtering the main json file

coco_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

Expected Directory:
script.py
COCO[
    annotations
    val2017
    train2017
]
"""

from pycocotools.coco import COCO
import requests
import os
from os.path import join
from tqdm import tqdm
import json
import re
def is_begining_with_upper(classe_name: str)-> bool:
    """"This function return if classe_name begin with upper letter"""

    pattern = "^[A-Z][a-zA-Z]*$"
    return bool(re.match(pattern, classe_name))

def is_class_in_azuria_zoo(classe_name: str)->bool:
    global  azuria_zoo_classes
    azuria_zoo_classes = {"Lying-Person":"Fall","Pederstrien":"Person"}
    if classe_name in azuria_zoo_classes:
        return True 
    return False
def classe_name_substitution(classe_name : str)-> str : 
     """This function transform classe_name to it's equivalent in azuria_zoo_classes"""

     if not is_begining_with_upper(classe_name):
         classe_name =  classe_name[0].upper() + classe_name[1:]
     if is_class_in_azuria_zoo(classe_name):
         classe_name = azuria_zoo_classes[classe_name]
     return classe_name 
class coco_category_filter:
    """
    Downloads images of one category & filters jsons
    to only keep annotations of this category
    """

    def __init__(self, json_path, _categ):
        self.coco = COCO(json_path)  # instanciate coco class
        self.categ = ''
        self.images = self.get_imgs_from_json(_categ)

    def get_imgs_from_json(self, _categ):
        """returns image names of the desired category"""
        # Get category ids
        self.catIds = self.coco.getCatIds(catNms=_categ)
        assert len(self.catIds) > 0, "[ERROR] cannot find category index for {}".format(_categ)
        print("catIds: ", self.catIds)
        # Get the corresponding image ids and images using loadImgs
        imgIds = []
        for c in self.catIds:
            imgIds += self.coco.getImgIds(catIds=c)  # get images over categories (logical OR)
        imgIds = list(set(imgIds))  # remove duplicates
        images = self.coco.loadImgs(imgIds)
        print(f"{len(images)} images of '{self.categ}' instances")
        return images

    def save_imgs(self, imgs_dir):
        """saves the images of this category"""
        print("Saving the images with required categories ...")
        os.makedirs(imgs_dir, exist_ok=True)
        # Save the images into a local folder
        for im in tqdm(self.images):
            img_data = requests.get(im['coco_url']).content
            with open(os.path.join(imgs_dir, im['file_name']), 'wb') as handler:
                handler.write(img_data)

    def filter_json_by_category(self, json_dir):
        """creates a new json with the desired category"""
        # {'supercategory': 'person', 'id': 1, 'name': 'person'}
        ### Filter images:
        print("Filtering the annotations ... ")
        imgs_ids = [x['id'] for x in self.images]  # get img_ids of imgs with the category (prefiltered)
        new_imgs = [x for x in self.coco.dataset['images'] if x['id'] in imgs_ids]  # select images by img_ids
        catIds = self.catIds
        ### Filter annotations
        new_annots = [x for x in self.coco.dataset['annotations'] if x['category_id'] in catIds]  # select annotations based on category id
        ### Reorganize the ids (note for reordering subset 1-N)
        new_imgs, annotations = self.modify_ids(new_imgs, new_annots)
        ### Filter categories
        new_categories = [x for x in self.coco.dataset['categories'] if x['id'] in catIds]
        print("new_categories: ", new_categories)
        data = {
            "images": new_imgs,
            "annotations": new_annots,
            "categories": new_categories
        }
        print("saving json: ")
        with open(json_dir, 'w') as f:
            json.dump(data, f)

    def modify_ids(self, images, annotations):
        """
        creates new ids for the images. I.e., maps existing image id to new subset image id and returns the dictionaries back
        images: list of images dictionaries

        images[n]['id']                                     # id of image
        annotations[n]['id']                                # id of annotation
        images[n]['id'] --> annotations[n]['image_id']      # map 'id' of image to 'image_id' in annotation
        """
        print("Reinitialicing images and annotation IDs ...")
        ### Images
        map_old_to_new_id = {}  # necessary for the annotations!
        for n, im in enumerate(images):
            map_old_to_new_id[images[n]['id']] = n + 1  # dicto with old im_ids and new im_ids
            images[n]['id'] = n + 1  # reorganize the ids
        ### Annotations
        for n, ann in enumerate(annotations):
            annotations[n]['id'] = n + 1
            old_image_id = annotations[n]['image_id']
            annotations[n]['image_id'] = map_old_to_new_id[old_image_id]  # replace im_ids in the annotations as well
        return images, annotations


def main(subset, year, root_dir, categories, experiment):
    json_file = "/home/mathis/Bureau/filter_categories/train.json"

    # Output files
    img_dir = join(root_dir, experiment, 'images')
    os.makedirs(img_dir, exist_ok=True)
    json_dir = join(root_dir, experiment, 'annotations')
    os.makedirs(json_dir, exist_ok=True)

    # Methods
    coco_filter = coco_category_filter(json_file, categories)  # instantiate class
    coco_filter.save_imgs(img_dir)
    coco_filter.filter_json_by_category(json_dir)


@logger.catch(reraise=True)
def coco_merge(
    input_extend: str,
    input_add: str,
    output_file: str,
    indent: Optional[int] = None,categories: list =['Smoke','Fire', 'Car', 'Truck','Bicycle'],
) -> str:
    """Merge COCO annotation files.

    Args:
        input_extend: Path to input file to be extended.
        input_add: Path to input file to be added.
        output_file : Path to output file with merged annotations.
        indent: Argument passed to `json.dump`. See https://docs.python.org/3/library/json.html#json.dump.
    """
    with open(input_extend, "r") as f:
        data_extend = json.load(f)
    with open(input_add, "r") as f:
        data_add = json.load(f)
    category_add =  data_add['categories']
    category_extend = data_extend['categories']
    for i in range(len(category_add)):
        if not is_begining_with_upper(category_add[i]['name']) or  is_class_in_azuria_zoo(category_add[i]['name']):
             category_add[i]['name']= classe_name_substitution(category_add[i]['name'])
    for i in range(len(category_extend)):
        if not is_begining_with_upper(category_extend[i]['name']) or  is_class_in_azuria_zoo(category_extend[i]['name']):
             category_extend[i]['name']= classe_name_substitution(category_extend[i]['name'])
    output: Dict[str, Any] = {
        k: data_extend[k] for k in data_extend if k not in ("images", "annotations")
    }

    output["images"], output["annotations"] = [], []

    for i, data in enumerate([data_extend, data_add]):
        logger.info(
            "Input {}: {} images, {} annotations".format(
                i + 1, len(data["images"]), len(data["annotations"])
            )
        )

        cat_id_map = {}
        for new_cat in data["categories"]:
            new_id = None
            for output_cat in output["categories"]:
                if new_cat["name"] == output_cat["name"]:
                    new_id = output_cat["id"]
                    break

            if new_id is not None:
                cat_id_map[new_cat["id"]] = new_id
            else:
                new_cat_id = max(c["id"] for c in output["categories"]) + 1
                cat_id_map[new_cat["id"]] = new_cat_id
                new_cat["id"] = new_cat_id
                output["categories"].append(new_cat)

        img_id_map = {}
        for image in data["images"]:
            n_imgs = len(output["images"])
            img_id_map[image["id"]] = n_imgs
            image["id"] = n_imgs

            output["images"].append(image)

        for annotation in data["annotations"]:
            n_anns = len(output["annotations"])
            annotation["id"] = n_anns
            annotation["image_id"] = img_id_map[annotation["image_id"]]
            annotation["category_id"] = cat_id_map[annotation["category_id"]]

            output["annotations"].append(annotation)

    logger.info(
        "Result: {} images, {} annotations".format(
            len(output["images"]), len(output["annotations"])
        )
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=indent, ensure_ascii=False)
    coco_filter = coco_category_filter(output_file, categories)
    coco_filter.filter_json_by_category(f"{output_file[:-4]}_1000.json")
    return output_file
