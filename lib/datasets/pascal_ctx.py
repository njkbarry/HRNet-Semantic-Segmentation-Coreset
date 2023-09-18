# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Referring to the implementation in
# https://github.com/zhanghang1989/PyTorch-Encoding
# ------------------------------------------------------------------------------

import os
import sys

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from collections import defaultdict

# sys.path.insert(0, os.getcwd())
# sys.path.insert(0, os.getcwd() + "/lib")

# from lib.datasets.base_dataset import BaseDataset

from .base_dataset import BaseDataset


class PASCALContext(BaseDataset):
    def __init__(
        self,
        root,
        list_path,
        num_samples=None,
        num_classes=59,
        multi_scale=True,
        flip=True,
        ignore_label=-1,
        base_size=520,
        crop_size=(480, 480),
        downsample_rate=1,
        scale_factor=16,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        super(PASCALContext, self).__init__(ignore_label, base_size, crop_size, downsample_rate, scale_factor, mean, std)

        self.root = os.path.join(root, "pascal_ctx/VOCdevkit/VOC2010")
        self.split = list_path

        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size

        # prepare data
        annots = os.path.join(self.root, "trainval_merged.json")
        img_path = os.path.join(self.root, "JPEGImages")
        from detail import Detail

        if "val" in self.split:
            self.detail = Detail(annots, img_path, "val")
            mask_file = os.path.join(self.root, "val.pth")
        elif "train" in self.split:
            self.mode = "train"
            self.detail = Detail(annots, img_path, "train")
            mask_file = os.path.join(self.root, "train.pth")
        else:
            raise NotImplementedError("only supporting train and val set.")
        self.files = self.detail.getImgs()

        # generate masks
        self._mapping = np.sort(
            np.array(
                [
                    0,
                    2,
                    259,
                    260,
                    415,
                    324,
                    9,
                    258,
                    144,
                    18,
                    19,
                    22,
                    23,
                    397,
                    25,
                    284,
                    158,
                    159,
                    416,
                    33,
                    162,
                    420,
                    454,
                    295,
                    296,
                    427,
                    44,
                    45,
                    46,
                    308,
                    59,
                    440,
                    445,
                    31,
                    232,
                    65,
                    354,
                    424,
                    68,
                    326,
                    72,
                    458,
                    34,
                    207,
                    80,
                    355,
                    85,
                    347,
                    220,
                    349,
                    360,
                    98,
                    187,
                    104,
                    105,
                    366,
                    189,
                    368,
                    113,
                    115,
                ]
            )
        )

        self._seg_index_to_name = {
            1: "aeroplane",
            2: "bicycle",
            3: "bird",
            4: "boat",
            5: "bottle",
            6: "bus",
            7: "car",
            8: "cat",
            9: "chair",
            10: "cow",
            11: "table",
            12: "dog",
            13: "horse",
            14: "motorbike",
            15: "person",
            16: "pottedplant",
            17: "sheep",
            18: "sofa",
            19: "train",
            20: "tvmonitor",
            21: "bag",
            22: "bed",
            23: "bench",
            24: "book",
            25: "building",
            26: "cabinet",
            27: "ceiling",
            28: "cloth",
            29: "computer",
            30: "cup",
            31: "door",
            32: "fence",
            33: "floor",
            34: "flower",
            35: "food",
            36: "grass",
            37: "ground",
            38: "keyboard",
            39: "light",
            40: "mountain",
            41: "mouse",
            42: "curtain",
            43: "platform",
            44: "sign",
            45: "plate",
            46: "road",
            47: "rock",
            48: "shelves",
            49: "sidewalk",
            50: "sky",
            51: "snow",
            52: "bedclothes",
            53: "track",
            54: "tree",
            55: "truck",
            56: "wall",
            57: "water",
            58: "window",
            59: "wood",
        }

        self._class_index_to_name = {
            1: "aeroplane",
            2: "bicycle",
            3: "bird",
            4: "boat",
            5: "bottle",
            6: "bus",
            7: "car",
            8: "cat",
            9: "chair",
            10: "cow",
            11: "diningtable",
            12: "dog",
            13: "horse",
            14: "motorbike",
            15: "person",
            16: "pottedplant",
            17: "sheep",
            18: "sofa",
            19: "train",
            20: "tvmonitor",
            21: "sky",
            22: "grass",
            23: "ground",
            24: "road",
            25: "building",
            26: "tree",
            27: "water",
            28: "mountain",
            29: "wall",
            30: "floor",
            31: "track",
            32: "keyboard",
            33: "ceiling",
        }

        self._key = np.array(range(len(self._mapping))).astype("uint8")

        print("mask_file:", mask_file)
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            self.masks = self._preprocess(mask_file)

        # Load index of image-wise class labels for MILO partitioning
        self.imagewise_labels = self._init_imagewise_label()

        self._init_name_to_index()  # Initialise dict

        # Initialise proportion dicts
        self._pixel_class_proportions = None
        self._occurence_class_proportions = None

    def get_occurence_class_proportion(self, index: int):
        if self._occurence_class_proportions is None:
            self._init_occurence_class_proportions()
        return self._occurence_class_proportions[index]

    def get_pixel_class_proportion(self, index: int):
        if self._pixel_class_proportions is None:
            self._init_pixel_class_proportions()
        return self._pixel_class_proportions[index]

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert values[i] in self._mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess(self, mask_file):
        masks = {}
        print("Preprocessing mask, this will take a while." + "But don't worry, it only run once for each split.")
        for i in range(len(self.files)):
            img_id = self.files[i]
            mask = Image.fromarray(self._class_to_index(self.detail.getMask(img_id)))
            masks[img_id["image_id"]] = mask
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        item = self.files[index]
        name = item["file_name"]
        img_id = item["image_id"]

        image = cv2.imread(os.path.join(self.detail.img_folder, name), cv2.IMREAD_COLOR)
        label = np.asarray(self.masks[img_id], dtype=np.int)
        size = image.shape

        if self.split == "val":
            image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            label = cv2.resize(label, self.crop_size, interpolation=cv2.INTER_NEAREST)
            label = self.label_transform(label)
        elif self.split == "testval":
            # evaluate model on val dataset
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            label = self.label_transform(label)
        else:
            image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def label_transform(self, label):
        if self.num_classes == 59:
            # background is ignored
            label = np.array(label).astype("int32") - 1
            label[label == -2] = -1
        else:
            label = np.array(label).astype("int32")
        return label

    def get_imagewise_label(self, name: str):
        if isinstance(name, str):
            v = name.replace(".jpg", "")
            imagewise_label = [k for k in self.class_lists_dict.keys() if v in self.class_lists_dict[k]]
            assert len(imagewise_label) != 0, "Image does not appear to have image-wise labels"
            if len(imagewise_label) > 1:
                # Return the least frequent class
                label_freqs = [len(self.class_lists_dict[k]) for k in imagewise_label]
                imagewise_label = [imagewise_label[np.argmin(np.asarray(label_freqs))]]
        elif isinstance(name, int):
            pass
        else:
            raise NotImplementedError
        return imagewise_label[0]

    def get_img_index(self, name: str):
        return self.name_to_index[name]

    def _init_name_to_index(
        self,
    ):
        self.name_to_index = {}
        for i in range(len(self.files)):
            item = self.files[i]
            self.name_to_index[item["file_name"]] = i

    def _init_imagewise_label(
        self,
    ):
        print("wait")
        class_lists_dir = os.path.join(self.root, "ImageSets/Main/")
        class_list_paths = [
            Path(class_lists_dir + "/" + p) for p in os.listdir(class_lists_dir) if self.split + ".txt" in p and p != self.split + ".txt"
        ]
        class_lists_dict = {}
        for class_list_path in class_list_paths:
            class_file = open(class_list_path, "r")
            class_list = class_file.read().split("\n")
            class_list = [l.replace("  1", "") for l in class_list if " 1" in l]
            class_name = class_list_path.stem.replace("_" + self.split, "")
            class_lists_dict[class_name] = class_list
            class_file.close()
        self.class_lists_dict = class_lists_dict

    def class_index_to_name(self, class_name: str, seg_class: bool = False):
        if seg_class:
            self._seg_index_to_name[class_name]
        else:
            self._class_index_to_name[class_name]

    def class_name_to_index(self, class_index: int, seg_class: bool = False):
        if seg_class:
            return list(self._seg_index_to_name.keys())[list(self._seg_index_to_name.values()).index(class_index)]
        else:
            return list(self._class_index_to_name.keys())[list(self._class_index_to_name.values()).index(class_index)]

    def _init_pixel_class_proportions(
        self,
    ):
        uniques = []
        counts = []
        for index, file in tqdm(enumerate(self), total=len(self), desc="Determining pixel-wise class proprotions"):
            image, label, size, name = file
            unique, count = np.unique(label, return_counts=True)
            uniques.append(unique)
            counts.append(count)
            index = self.get_img_index(name)
            assert name == self.__getitem__(index)[3], "Wrong index"
        pixel_df = pd.DataFrame({"class": np.concatenate(uniques), "count": np.concatenate(counts)})
        # pixel_df = pixel_df.groupby("class").sum()
        pixel_df = pixel_df.groupby("class").mean()
        pixel_df = pixel_df.drop(index=-1)
        # pixel_df = pixel_df / pixel_df.sum()
        self._pixel_df = pixel_df
        pixel_class_proportions = (pixel_df / pixel_df.sum())["count"].to_dict()
        pixel_class_proportions[-1] = "Nan"
        self._pixel_class_proportions = pixel_class_proportions

    def _init_occurence_class_proportions(
        self,
    ):
        uniques = []
        counts = []
        for index, file in tqdm(enumerate(self), total=len(self), desc="Determining occurence class proprotions"):
            image, label, size, name = file
            unique, _ = np.unique(label, return_counts=True)
            uniques.append(unique)
            counts.append(np.ones_like(unique))
            index = self.get_img_index(name)
            assert name == self.__getitem__(index)[3], "Wrong index"
        pixel_df = pd.DataFrame({"class": np.concatenate(uniques), "count": np.concatenate(counts)})
        pixel_df = pixel_df.groupby("class").sum()
        pixel_df = pixel_df.drop(index=-1)
        pixel_df = pixel_df / pixel_df.sum()
        self._occurence_df = pixel_df
        occurence_class_proportions = (pixel_df / pixel_df.sum())["count"].to_dict()
        occurence_class_proportions[-1] = "Nan"
        self._occurence_class_proportions = occurence_class_proportions

    def _init_co_ocurrence_df(
        self,
    ):
        uniques = []
        counts = []
        for index, file in tqdm(enumerate(self), total=len(self), desc="Determining co-occurence class proprotions"):
            image, label, size, name = file
            unique, _ = np.unique(label, return_counts=True)
            uniques.append(unique)
            counts.append(np.ones_like(unique))
            index = self.get_img_index(name)
            assert name == self.__getitem__(index)[3], "Wrong index"
        od = defaultdict(list)
        for i in range(len(uniques)):
            for s in set(uniques[i]):
                od[s].append(i)
        x = [(k1, k2, len(set(d1) & set(d2))) for k1, d1 in od.items() for k2, d2 in od.items()]
        self._co_occurences_df = pd.DataFrame(x).pivot(index=0, columns=1, values=2)


if __name__ == "__main__":
    # DEV TESTING
    trainset = PASCALContext(
        root="/home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/data/",
        list_path="val",
        num_samples=None,
        num_classes=59,
        multi_scale=True,
        flip=True,
        ignore_label=-1,
        base_size=520,
        crop_size=(520, 520),
        downsample_rate=1,
        scale_factor=16,
    )

    # trainset.get_pixel_class_proportion(-1)
    # trainset._init_co_ocurrence_df()

    # Figure 4.2.7
    # trainset._co_occurences_df.to_csv(
    #     "/home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/analysis/plot_csv/test/section_4/figure_4_2_7_val.csv"
    # )

    # Figfure 4.2.4append
    # df.to_csv(
    #     "/home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/analysis/plot_csv/test/section_4/figure_4_2_4_val.csv"
    # )

    # Figure 4.2.1
    # df = pd.concat([trainset._pixel_df, trainset._occurence_df], axis=1)
    # df = df.set_axis(["pixel-wise", "occurence"], axis=1)
    # df.to_csv(
    #     "/home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/analysis/plot_csv/test/section_4/figure_4_2_1_val.csv"
    # )

    image, label, size, name = trainset.__getitem__(10)
    values, counts = np.unique(label, return_counts=True)
    index = np.nanargmin([trainset.get_pixel_class_proportion(c) for c in values])
    print("break point")
