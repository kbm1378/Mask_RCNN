# coco.py 에서 training 부분만 빼서 커스텀
import os
import sys
import time
import numpy as np
import imgaug  
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
import urllib.request
import shutil
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  
from mrcnn.config import Config
from mrcnn import model as modellib, utils
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
from coco import CocoDataset

############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    NAME = "coco"
    # GPU 개수와 한 GPU가 한번에 분석할 이미지 개수 설정하는 부분  (GPU 12GB인 경우 2로 했다고함)
    IMAGES_PER_GPU = 2
    # GPU 개수 (default = 1)
    # GPU_COUNT = 8
    # 분류할 클래스 개수 (배경 포함)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    # 그 외에도 mrcnn/config.py 참고하여 다양한 튜닝 관련 변수 오버라이딩 가능함. 
    # ex: STEPS_PER_EPOCH, BACKBONE, LEARNING_RATE, WEIGHT_DECAY 등..


############################################################
#  Training
############################################################


if __name__ == '__main__':    
    # train시에 epoch 마다 모델 아키텍쳐/가중치 파일을 저장하는 경로 (변경 가능)
    MODEL_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    
    # Configurations & Create model
    config = CocoConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_LOGS_DIR)

    # Pretrained model (.h5 파일) 로 어떤 것을 사용할지? (아래 4가지 중 1가지 주석 해제하고 사용)
    PRETRAINED_MODEL_PATH = COCO_MODEL_PATH # coco 학습 가중치 사용
    # PRETRAINED_MODEL_PATH = model.find_last() # 가장 최근 학습한 모델
    # PRETRAINED_MODEL_PATH = model.get_imagenet_weights() # 이미지넷 가중치 사용
    # PRETRAINED_MODEL_PATH = "/path/to/weight.h5" # 직접 설정 
    model.load_weights(PRETRAINED_MODEL_PATH, by_name=True)

    # Training & Validation 할 데이터 경로 
    # Coco 포맷으로 해야 하며 라벨링이 기록된 coco 포맷 json 파일 경로와 
    # 이미지들이 들어있는 디렉토리 경로를 넣어줘야 함
    TRAINING_JSON_PATH = "/path/to/json"
    TRAINING_IMAGE_DIR = "/path/to/image/dir"
    VALIDATION_JSON_PATH = "/path/to/json"
    VALIDATION_IAMGE_DIR = "/path/to/image/dir"
    dataset_train = CocoDataset()
    dataset_train.load_coco_custom(TRAINING_JSON_PATH, TRAINING_IMAGE_DIR)
    dataset_train.prepare()
    dataset_val = CocoDataset()
    dataset_val.load_coco_custom(VALIDATION_JSON_PATH, VALIDATION_IMAGE_DIR)
    dataset_val.prepare()

    # Image Augmentation : Right/Left flip 50% of the time -> 변경 가능 
    augmentation = imgaug.augmenters.Fliplr(0.5)


    # 아래 모든 트레이닝 코드는 예시로 변경 가능함. 
    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation)
    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)

    