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

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, sys
import pandas as pd
import time
import copy

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Connect your script to Neptune
import neptune
import neptune_config

from detectron2.structures import BoxMode

from Trainer import Trainer


def main():
    data_dir = "../data/"

    train_df = pd.read_csv(os.path.join(data_dir, "augmented8.csv"))
    # train_df = pd.read_csv(os.path.join(data_dir, "train_df_modified.csv"))
    # train_df = train_df.drop(error_list)

    keypoint_names = train_df.columns.to_list()[1:]
    keypoint_flip_map = [
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_palm", "right_palm"),
        ("left_instep", "right_instep"),
    ]
    # for i in range(0, len(keypoint_names), 2):
    #     keypoint_flip_map.append((keypoint_names[i], keypoint_names[i+1]))
    # ((nose_x, nose_y), (right_arm_x, right_arm_y))
    columns = train_df.columns[1:].to_list()[::2]
    keypoint_names = [
        label.replace("_x", '').replace("_y", '') for label in columns
    ] # 24 keypoints name (nose, right_arm)

    imgs = train_df.iloc[:, 0].to_numpy() # image path
    keypoints = train_df.iloc[:, 1:].to_numpy() # 24 * 2 keypoints
    imgs_train, imgs_val, keypoints_train, keypoints_val = \
        train_val_split(imgs, keypoints, random_state=42)

    # imgs_train, imgs_val, keypoints_train, keypoints_val = \
        # train_val_split2(aug_df, train_df)

    imgs_d = {
        "train": imgs_train,
        "val": imgs_val
    }
    keypoints_d = {
        "train": keypoints_train,
        "val": keypoints_val
    }

    for d in ["train", "val"]:
        DatasetCatalog.register(
            "keypoints_" + d,
            lambda d=d: get_data_dicts(
                data_dir, imgs_d[d], keypoints_d[d], phase=d
            )
        )
        MetadataCatalog.get("keypoints_" + d).set(
            thing_classes=["human"]
        )
        MetadataCatalog.get("keypoints_" + d).set(
            keypoint_names=keypoint_names
        )
        MetadataCatalog.get("keypoints_" + d).set(
            keypoint_flip_map=keypoint_flip_map
        )
        MetadataCatalog.get("keypoints_" + d).set(
            evaluator_type="coco"
        )

    motions_metadata = MetadataCatalog.get("keypoints_train")
    print(motions_metadata)
    ns = neptune.init(
        project_qualified_name="mybirth0407/dacon-motion",
        api_token=neptune_config.token
    )

    # keypoint_rcnn_R_50_FPN_3x.yaml
    # keypoint_rcnn_X_101_32x8d_FPN_3x.yaml
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("keypoints_train",)
    cfg.DATASETS.TEST = ("keypoints_val",)
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = (5000, 7000, 9000)         # do not decay learning rate
    # The iteration number to decrease learning rate by GAMMA.
    cfg.SOLVER.GAMMA = 0.1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 24
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((24, 1), dtype=float).tolist()
    # cfg.TEST.EVAL_PERIOD = 1000

    # # Create experiment
    neptune.create_experiment(f"Detectron2")

    neptune.log_metric("ims_per_batch", cfg.SOLVER.IMS_PER_BATCH)
    neptune.log_metric("learning_rate", cfg.SOLVER.BASE_LR)
    neptune.log_metric("batch_per_image", cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE)
    neptune.log_metric("num_epochs", cfg.SOLVER.MAX_ITER)
    neptune.log_metric("augmentation", 8)
    counter = ns._get_current_experiment()._id

    cfg.OUTPUT_DIR = f"../outputs/{counter}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    test_dir = os.path.join(data_dir, "test_imgs")
    test_list = os.listdir(test_dir)
    test_list.sort()
    except_list = []

    files = []
    preds = []
    for file in tqdm(test_list):
        filepath = os.path.join(test_dir, file)
        # print(filepath)
        im = cv2.imread(filepath)
        outputs = predictor(im)
        outputs = outputs["instances"].to("cpu").get("pred_keypoints").numpy()
        files.append(file)
        pred = []
        try:
            for out in outputs[0]:
                pred.extend([float(e) for e in out[:2]])
        except:
            pred.extend([0] * 48)
            except_list.append(filepath)
            # print(filepath)
        preds.append(pred)

    df_sub = pd.read_csv(f"../data/sample_submission.csv")
    df = pd.DataFrame(columns=df_sub.columns)
    df["image"] = files
    df.iloc[:, 1:] = preds

    df.to_csv(f"../submissions/{counter}.csv", index=False)
    print(except_list)


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
            if "Origin" in key and i in r:
                validations.append(np.where(imgs == value[i])[0][0])
            else:
                trains.append(np.where(imgs == value[i])[0][0])
    return (
        imgs[trains], imgs[validations],
        keypoints[trains], keypoints[validations]
    )


def train_val_split2(augmented, train):
    train_imgs = train.iloc[:, 0].to_numpy()
    train_keypoints = train.iloc[:, 1:].to_numpy()
    aug_imgs = augmented.iloc[:, 0].to_numpy()
    aug_keypoints = augmented.iloc[:, 1:].to_numpy()
    return aug_imgs, train_imgs, aug_keypoints, train_keypoints


def get_data_dicts(data_dir, imgs, keypoints, phase):
    # train_dir = os.path.join(data_dir, "augmented" if phase=="train" else "train_imgs")
    train_dir = os.path.join(data_dir, "augmented8")
    # train_dir = os.path.join(data_dir, "train_imgs")
    dataset_dicts = []

    for idx, item in tqdm(enumerate(zip(imgs, keypoints))):
        img, keypoint = item[0], item[1]

        record = {}
        filepath = os.path.join(train_dir, img)
        record["height"], record["width"] = cv2.imread(filepath).shape[:2]
        record["file_name"] = filepath
        record["image_id"] = idx

        keypoints_v = []
        flag = True
        for i, keypoint_ in enumerate(keypoint):
            keypoints_v.append(keypoint_) # if coco set, should be added 0.5
            if keypoint_ < 0:
                flag = False
            if i % 2 == 1:
                if flag:
                    keypoints_v.append(2)
                else:
                    keypoints_v.append(0)
                flag = True


        x = keypoint[0::2]
        y = keypoint[1::2]
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        obj = {
            "bbox": [x_min, y_min, x_max, y_max],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,
            "keypoints": keypoints_v
        }

        record["annotations"] = [obj]
        dataset_dicts.append(record)
    return dataset_dicts


if __name__ == '__main__':
    main()
