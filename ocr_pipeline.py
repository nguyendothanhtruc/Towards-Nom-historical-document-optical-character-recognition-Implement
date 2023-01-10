import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import yaml
import dynamic_yaml
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm
import pprint

import utils

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
# tf.compat.v1.disable_eager_execution()

# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     print("Limited GPU")
# except:
#     print("Failed to limit GPU")

def run_pipeline(args):
    # ================================CONFIG OCR PIPELINE================================
    with open(args.config) as fileobj:
        config = dynamic_yaml.load(fileobj)

    config = edict(config)
    pprint.pprint(config)

    if not os.path.exists(config.BENCHMARK_PATH_MB):
        os.makedirs(config.BENCHMARK_PATH_MB, exist_ok=True)

    if not os.path.exists(config.BENCHMARK_PATH):
        os.makedirs(config.BENCHMARK_PATH, exist_ok=True)

    OCR_RESULTS_CSV_MB = os.path.join(config.BENCHMARK_PATH_MB,
                              f"ocr_results_{config.MODEL_NAME}.csv")

    OCR_RESULTS_WO_MB_CSV = os.path.join(config.BENCHMARK_PATH,
                          f"ocr_results_{config.MODEL_NAME}_wo_MacBert.csv")

    # OCR_RESULTS_PKL = os.path.join(config.BENCHMARK_PATH,
    #                           f"ocr_results_{config.MODEL_NAME}.pkl")

    if not os.path.isfile(OCR_RESULTS_CSV_MB):
        # ================================PREPROCESSING CHARACTERS CLASSIFICATION================================
        # detect_results -> col: conf, x1, x2, y1, y2, source_img
        detect_results = utils.preprocess_classifier(config)
                          
        # Crop characters for classifying by RESNET101
        # detect_results -> col: conf, x1, x2, y1, y2, source_img, crop_img
        detect_results = utils.crop_chars(config, detect_results)
        # ================================CLASSIFICATION PHASE================================
        # Load RESNET101 model
        RESNET = tf.keras.models.load_model(config.RESNET_PATH, compile=False)

        # Classify
        ocr_results = utils.classify_characters(config, RESNET, detect_results)
        
        # ================================POSTPROCESSING PHASE================================
        # Create results folder, storing benchmark results and ocr files
        if not os.path.exists(config.RESULT_OCR_PATH):
            os.mkdir(config.RESULT_OCR_PATH)

        # Create results folder, storing benchmark results and ocr files
        if not os.path.exists(config.RESULT_OCR_PATH_MB):
            os.mkdir(config.RESULT_OCR_PATH_MB)
            
        # Sort OCR results according to order (TTB - RTL)
        sorted_ocr_results, sep_lst = utils.post_processing_ocr(config, ocr_results)
        
        #With and without MacBert
        sorted_ocr_results_wo_MB = sorted_ocr_results.copy()

        # Correct OCR results with MacBert
        sorted_ocr_results = utils.run_macbert(config, sorted_ocr_results, sep_lst)

        # Save sorted ocr results
        sorted_ocr_results.to_csv(OCR_RESULTS_CSV_MB, index=False)
        sorted_ocr_results_wo_MB.to_csv(OCR_RESULTS_WO_MB_CSV, index = False)
        #sorted_ocr_results.to_pickle(OCR_RESULTS_PKL)
        
        # Write ocr results to file
        utils.write_final_ocr_results(config.RESULT_OCR_PATH, sorted_ocr_results_wo_MB, sep_lst)
        utils.write_final_ocr_results(config.RESULT_OCR_PATH_MB, sorted_ocr_results, sep_lst)

        # ================================PREPROCESS BENCHMARK PHASE================================
        ######### OCR WITH MACBERT
        # Create groundtruth folder, storing txt files
        if not os.path.exists(config.GT_PATH_MB):
            os.mkdir(config.GT_PATH_MB)

        GT_DS = pd.read_csv(config.GT_DS_PATH)
        utils.create_gt_files(config.GT_PATH_MB, GT_DS)

        # Create detection folder, storing txt files
        if not os.path.exists(config.DT_PATH_MB):
            os.mkdir(config.DT_PATH_MB)

        DT_DS_MB = pd.read_csv(OCR_RESULTS_CSV_MB)
        utils.create_dt_files(config.DT_PATH_MB, DT_DS_MB)

        # Create results folder, storing benchmark files
        if not os.path.exists(config.RESULT_PATH_MB):
            os.mkdir(config.RESULT_PATH_MB)

        ######### OCR WITHOUT MACBERT
        # Create groundtruth folder, storing txt files
        if not os.path.exists(config.GT_PATH):
            os.mkdir(config.GT_PATH)

        GT_DS = pd.read_csv(config.GT_DS_PATH)
        utils.create_gt_files(config.GT_PATH, GT_DS)

        # Create detection folder, storing txt files
        if not os.path.exists(config.DT_PATH):
            os.mkdir(config.DT_PATH)

        DT_DS = pd.read_csv(OCR_RESULTS_WO_MB_CSV)
        utils.create_dt_files(config.DT_PATH, DT_DS)

        # Create results folder, storing benchmark files
        if not os.path.exists(config.RESULT_PATH):
            os.mkdir(config.RESULT_PATH)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file", default="/data4/trucndt3/OCR/ocr_config.yaml"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_pipeline(args)
