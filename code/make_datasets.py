# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data as data_utils
from torchvision import datasets, models, transforms

from sklearn.model_selection import KFold, train_test_split

import albumentations as A
import albumentations_experimental as AE
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm


def main():
    data_dir = "../data/"
    train_dir = os.path.join(data_dir, "train_imgs")
    train_df = pd.read_csv(os.path.join(data_dir, "train_df_modified.csv"))

    keypoint_names = train_df.columns.to_list()[1:]
    keypoint_flip_map = []
    for i in range(0, len(keypoint_names) // 2, 2):
        keypoint_flip_map.append((keypoint_names[i], keypoint_names[i+1]))

    columns = train_df.columns[1:].to_list()[::2]
    # class_labels
    keypoints_names = [
        label.replace("_x", '').replace("_y", '') for label in columns
    ]

    imgs = train_df.iloc[:, 0].to_numpy()
    keypoints = train_df.iloc[:, 1:].to_numpy()

    pair_keypoints = []
    for keypoint in keypoints:
        a_keypoints = []
        for i in range(0, keypoint.shape[0], 2):
            a_keypoints.append((float(keypoint[i]), float(keypoint[i+1])))
        pair_keypoints.append(a_keypoints)
    pair_keypoints = np.array(pair_keypoints)

    # imgs_train, imgs_val, keypoints_train, keypoints_val = \
    #     train_val_split(imgs, pair_keypoints, random_state=42)

    aug_dir = os.path.join(data_dir, "augmented8")
    os.makedirs(aug_dir, exist_ok=True)

    aug_img_list = []
    aug_keypoints_list = []
    for img, keypoints_ in tqdm(zip(imgs, pair_keypoints)):
        keypoint_params = A.KeypointParams(
            format="xy", label_fields=["class_labels"],
            remove_invisible=False, angle_in_degrees=True
        )

        ori_img = cv2.imread(os.path.join(train_dir, img))

        transform_list = [
            A.Compose([
                A.RandomCrop(height=1080, width=1920, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.RandomCrop(height=720, width=960, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.RandomCrop(height=540, width=720, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.RandomCrop(height=960, width=960, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.CenterCrop(height=720, width=1280, p=1)
            ], keypoint_params=keypoint_params),

            A.Compose([
                A.Rotate(limit=45, p=1),
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=45, p=1),
                A.RandomCrop(height=720, width=960, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=45, p=1),
                A.RandomCrop(height=540, width=720, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=45, p=1),
                A.RandomCrop(height=960, width=960, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=45, p=1),
                A.CenterCrop(height=720, width=1280, p=1)
            ], keypoint_params=keypoint_params),

            A.Compose([
                A.Rotate(limit=60, p=1),
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=60, p=1),
                A.RandomCrop(height=720, width=960, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=60, p=1),
                A.RandomCrop(height=540, width=720, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=60, p=1),
                A.RandomCrop(height=960, width=960, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=60, p=1),
                A.CenterCrop(height=720, width=1280, p=1)
            ], keypoint_params=keypoint_params),

            A.Compose([
                A.RandomScale(scale_limit=0.3, p=1),
                A.RandomCrop(height=720, width=960, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.RandomScale(scale_limit=0.3, p=1),
                A.RandomCrop(height=540, width=720, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.RandomScale(scale_limit=0.3, p=1),
                A.RandomCrop(height=720, width=1280, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.RandomScale(scale_limit=0.3, p=1),
                A.CenterCrop(height=720, width=1280, p=1)
            ], keypoint_params=keypoint_params),

            A.Compose([
                A.Rotate(limit=45, p=1),
                A.RandomScale(scale_limit=0.3, p=1),
                A.RandomCrop(height=720, width=960, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=45, p=1),
                A.RandomScale(scale_limit=0.3, p=1),
                A.RandomCrop(height=540, width=720, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=45, p=1),
                A.RandomScale(scale_limit=0.3, p=1),
                A.RandomCrop(height=720, width=1280, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=45, p=1),
                A.RandomScale(scale_limit=0.3, p=1),
                A.CenterCrop(height=720, width=1280, p=1)
            ], keypoint_params=keypoint_params),

            A.Compose([
                A.Rotate(limit=60, p=1),
                A.RandomScale(scale_limit=0.3, p=1),
                A.RandomCrop(height=720, width=960, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=60, p=1),
                A.RandomScale(scale_limit=0.3, p=1),
                A.RandomCrop(height=540, width=720, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=60, p=1),
                A.RandomScale(scale_limit=0.3, p=1),
                A.RandomCrop(height=720, width=1280, p=1)
            ], keypoint_params=keypoint_params),
            A.Compose([
                A.Rotate(limit=60, p=1),
                A.RandomScale(scale_limit=0.3, p=1),
                A.CenterCrop(height=720, width=1280, p=1)
            ], keypoint_params=keypoint_params)
        ]

        """
        RandomCrop_1
        A.RandomCrop(height=720, width=960, p=1)
        RandomCrop_2
        A.RandomCrop(height=540, width=720, p=1)
        RandomCrop_3
        A.RandomCrop(height=960, width=960, p=1)
        CenterCrop
        A.CenterCrop(height=720, width=1280, p=1)
        """

        # Ver 1
        # transform_name = [
        #     "Origin", "RandomCrop_1", "RandomCrop_2", "RandomCrop_3", "CenterCrop",
        #     "Rotate45", "Rotate45_RandomCrop_1", "Rotate45_RandomCrop_2", "Rotate45_RandomCrop_3", "Rotate45_CenterCrop",
        #     "Resize", "Resize_RandomCrop_1", "Resize_RandomCrop_2", "Resize_RandomCrop_3", "Resize_CenterCrop"
        # ]

        # # Ver 2
        # transform_name = [
        #     "Origin", "HFlip"
        # ]

        # # Ver 3
        # transform_name = [
        #     "Origin", "RandomCrop_1", "RandomCrop_2", "CenterCrop",
        #     "Rotate45"
        # ]

        # # Ver 4
        # transform_name = [
        #     "Origin", "RandomCrop_1", "RandomCrop_2", "RandomCrop_3", "CenterCrop",
        #     "Rotate45", "Rotate45_RandomCrop_1", "Rotate45_RandomCrop_2", "Rotate45_RandomCrop_3", "Rotate45_CenterCrop"
        # ]

        # # Ver 5
        # transform_name = [
        #     "Origin", "RandomCrop_1", "RandomCrop_2", "RandomCrop_3", "CenterCrop",
        #     "Rotate45", "Rotate45_RandomCrop_1", "Rotate45_RandomCrop_2", "Rotate45_RandomCrop_3", "Rotate45_CenterCrop",
        #     "Rotate60", "Rotate60_RandomCrop_1", "Rotate60_RandomCrop_2", "Rotate60_RandomCrop_3", "Rotate60_CenterCrop",
        # ]

        # # Ver 6
        # transform_name = [
        #     "Origin", "RandomCrop_1", "RandomCrop_2", "RandomCrop_3", "CenterCrop",
        #     "Rotate45", "Rotate45_RandomCrop_1", "Rotate45_RandomCrop_2", "Rotate45_RandomCrop_3", "Rotate45_CenterCrop",
        #     "Rotate90", "Rotate90_RandomCrop_1", "Rotate90_RandomCrop_2", "Rotate90_RandomCrop_3", "Rotate90_CenterCrop"
        # ]

        # Ver 7
        transform_name = [
            "Origin", "RandomCrop_1", "RandomCrop_2", "RandomCrop_3", "CenterCrop",
            "Rotate45", "Rotate45_RandomCrop_1", "Rotate45_RandomCrop_2", "Rotate45_RandomCrop_3", "Rotate45_CenterCrop",
            "Rotate60", "Rotate60_RandomCrop_1", "Rotate60_RandomCrop_2", "Rotate60_RandomCrop_3", "Rotate60_CenterCrop",
            "Rescale_RandomCrop_1", "Rescale_RandomCrop_2", "Rescale_RandomCrop_3", "Rescale_CenterCrop",
            "Rescale_Rotate45_RandomCrop_1", "Rescale_Rotate45_RandomCrop_2", "Rescale_Rotate45_RandomCrop_3", "Rescale_Rotate45_CenterCrop",
            "Rescale_Rotate60_RandomCrop_1", "Rescale_Rotate60_RandomCrop_2", "Rescale_Rotate60_RandomCrop_3", "Rescale_Rotate60_CenterCrop",
        ]

        # Ver 8
        transform_name = [
            "Origin", "RandomCrop_1", "RandomCrop_2", "RandomCrop_3", "CenterCrop",
            "Rotate45", "Rotate45_RandomCrop_1", "Rotate45_RandomCrop_2", "Rotate45_RandomCrop_3", "Rotate45_CenterCrop",
            "Rescale_RandomCrop_1", "Rescale_RandomCrop_2", "Rescale_RandomCrop_3", "Rescale_CenterCrop",
            "Rescale_Rotate45_RandomCrop_1", "Rescale_Rotate45_RandomCrop_2", "Rescale_Rotate45_RandomCrop_3", "Rescale_Rotate45_CenterCrop",
        ]


        for name, transform in zip(transform_name, transform_list):
            augmented = transform(
                image=ori_img,
                keypoints=keypoints_,
                class_labels=keypoints_names
            )
            aug_img = augmented["image"]
            aug_keypoints = augmented["keypoints"]
            aug_keypoints = np.array(aug_keypoints).flatten()
            aug_savename = f"{name}_{img}"

            cv2.imwrite(os.path.join(aug_dir, aug_savename), aug_img)

            aug_img_list.append(aug_savename)
            aug_keypoints_list.append(aug_keypoints)

    df_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    df = pd.DataFrame(columns=df_sub.columns)
    df["image"] = aug_img_list
    df.iloc[:, 1:] = aug_keypoints_list
    df.to_csv(os.path.join(data_dir, "augmented8.csv"), index=False)


def train_val_split(imgs, keypoints, random_state=42):
    d = dict()
    for file in imgs:
        key = ''.join(file.split('-')[:-1])
        if key not in d.keys():
            d[key] = [file]
        else:
            d[key].append(file)
            
    np.random.seed(random_state)
    trains = []
    validations = []
    for key, value in d.items():
        r = np.random.randint(len(value), size=2)
        for i in range(len(value)):
            if i in r:
                validations.append(np.where(imgs == value[i])[0][0])
            else:
                trains.append(np.where(imgs == value[i])[0][0])
    return (
        imgs[trains], imgs[validations],
        keypoints[trains], keypoints[validations]
    )

if __name__ == '__main__':
    main()