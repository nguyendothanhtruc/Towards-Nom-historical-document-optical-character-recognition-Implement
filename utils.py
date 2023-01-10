import sys
sys.path.append('/content/gdrive/MyDrive/Intro_NLP/Code/OCR/macBERT')

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

import os
import numpy as np
import pandas as pd
from PIL import Image
from imgaug import augmenters as iaa
from tqdm import tqdm

import post_processing
import pre_processing
import macbert_correction

#================================CHARACTERS CLASSIFICATION PHASE================================
def preprocess_classifier(config):
    print("Process: Create bounding box dataframe")
    
    detect_results = pd.DataFrame([], columns=["conf", "x1", "x2", "y1", "y2"])
    source_img = []

    img_files = [os.path.join(config.DATA_PATH, f) for f in sorted(os.listdir(config.DATA_PATH))]
    label_files = [os.path.join(config.YOLO_TEST_LABELS_PATH, f) for f in sorted(os.listdir(config.YOLO_TEST_LABELS_PATH))]

    for img_path, label_path in zip(img_files, label_files):
        
        char_detect_pp = pre_processing.create_bbox_df(img_path,label_path)
        
        detect_results = pd.concat([detect_results, char_detect_pp])
        source_img.extend([img_path.split("/")[-1]]*len(char_detect_pp))
        
    detect_results["source_img"] = source_img
    print("Process done!")
    
    return detect_results
  
def crop_chars(config, detect_results):
    print("Process: Crop characters")
    if not os.path.exists(config.YOLO_TEST_CROPS_PATH):
        os.makedirs(config.YOLO_TEST_CROPS_PATH, exist_ok=True)
        
    crop_filename = []
    img_files = detect_results["source_img"].unique()
    if len(os.listdir(config.YOLO_TEST_CROPS_PATH)) == 0:
        with tqdm(total = len(img_files)) as pbar:
            for img in img_files:
                characters = detect_results[detect_results["source_img"] == img].reset_index(drop = True)
                original = Image.open(os.path.join(config.DATA_PATH, img))
                
                for index, bbox in characters.iterrows():
                    cropped_char = original.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                    
                    origin_file_name = bbox.source_img.split(".")[0]
                    file_name = f"{origin_file_name}_{index}.jpg"
                    cropped_char.save(os.path.join(config.YOLO_TEST_CROPS_PATH, file_name))
                    crop_filename.append(file_name)

                pbar.update()

    else:
        with tqdm(total = len(img_files)) as pbar:
            for img in img_files:
                characters = detect_results[detect_results["source_img"] == img].reset_index(drop = True)            
                for index, bbox in characters.iterrows():                    
                    origin_file_name = bbox.source_img.split(".")[0]
                    file_name = f"{origin_file_name}_{index}.jpg"
                    crop_filename.append(file_name)

                pbar.update()
                                       
    detect_results["crop_img"] = crop_filename
    print("Process done!")

    return detect_results

def decode_images(config, char_batch):
    pad_modes = ["constant", "maximum", "mean", "median", "minimum"]
    main_aug_val = iaa.Sequential(
    [
        iaa.Grayscale(alpha=1.0),
        iaa.PadToSquare(pad_mode=pad_modes, pad_cval=(0, 255)),
        iaa.Resize({"height": int(config.INPUT_SHAPE), "width": int(config.INPUT_SHAPE)})
    ])
    char_batch = main_aug_val.augment_images(char_batch)
    return char_batch
        
def classify_characters(config, model, detect_results):
    print("Process: Classify characters")
    
    label_lst = []
    score_lst = []
    candidate_lst = []
    x1_lst = []
    x2_lst = []
    y1_lst = []
    y2_lst = []
    conf_lst = []
    scr_img_lst = []
    
    img_files = detect_results["source_img"].unique()
    
    TRAIN_DS = pd.read_csv(config.TRAIN_DS_PATH)
    MAP_LABEL_DICT = TRAIN_DS["label"].unique()
    PRED_THRES = float(config.THRESHOLD)
    TOP_K = int(config.TOP_K)
    
    with tqdm(total = len(img_files)) as pbar:
        for img in img_files:
            characters = detect_results[detect_results["source_img"] == img]
            
            char_batch = []
            for _, char in characters.iterrows():
                char_img = np.array(Image.open(os.path.join(config.YOLO_TEST_CROPS_PATH, char["crop_img"])).convert(mode = "RGB"))
                char_batch.append(char_img)

            char_batch = decode_images(config, char_batch)
            char_batch = np.array(char_batch)
            
            predicts = model.predict(char_batch, verbose = 0)
            score_idx_batch = np.argsort(predicts, axis=1)[:,-TOP_K:]
     
            candidate_batch = []
            score_batch = []
            for idx in range(len(score_idx_batch)):
                candidate_batch.append(np.array(MAP_LABEL_DICT[score_idx_batch[idx, :]])[::-1])
                score_batch.append(predicts[idx, score_idx_batch[idx]][::-1])
                
            label_batch = np.array(MAP_LABEL_DICT[np.argmax(predicts, axis =1)])

            label_lst.extend(label_batch)
            score_lst.extend(score_batch)
            candidate_lst.extend(candidate_batch)
            
            pbar.update() 

    print("Process done!")
    
    detect_results["label"] = label_lst
    detect_results["candidate"] = candidate_lst
    detect_results["score_cls"] = score_lst
    return detect_results

    return recognized_results

#================================POSTPROCESSING PHASE================================
def post_processing_ocr(config, df):
    print("Process: Sort OCR results")
    
    bbox_df = []
    source_img_lst = df["source_img"].unique()
    sep_lst = []
    
    with tqdm(total = len(source_img_lst)) as pbar:
        for source_img in source_img_lst:
            img_df = df[df['source_img'] == source_img]
            listOfBBoxes = post_processing.createListOfBBoxes(img_df)
            listOfSortedBBoxes = post_processing.SortBBoxRTLAndTTB(listOfBBoxes)
            
            sorted_bbox_lst = []
            
            
            for bbox in listOfSortedBBoxes:
                sorted_bbox_lst.append(np.array([int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2), float(bbox.x_center), float(bbox.y_center)])) 
                bbox_df.append(np.array([bbox.label, bbox.conf_score, int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2), bbox.candidate_lst, bbox.score_cls_lst, source_img]))
            
            sorted_bbox_lst = np.array(sorted_bbox_lst)
            
            
            sep_img_lst = post_processing.SplitLine(sorted_bbox_lst, hyper_lambda=1.5)
            sep_lst.append(sep_img_lst)

            pbar.update()
    
    bbox_df = np.array(bbox_df)
    df = pd.DataFrame({
    "label": bbox_df[:, 0],
    "conf": bbox_df[:, 1],
    "x1": bbox_df[:, 2],
    "y1": bbox_df[:, 3],
    "x2": bbox_df[:, 4],
    "y2": bbox_df[:, 5],
    "candidate": bbox_df[:, 6],
    "score_cls": bbox_df[:, 7],
    "source_img": bbox_df[:, 8]
    }) 
    
    print("Process done!")
    
    return df, sep_lst

def get_code_utf16(char):
    char = str(char)
    char = char.split("\\U")[1]
    index = 0
    for i in range(0, len(char)):
        if not char[i] == '0':
            index = i
            break
    return char[index:-1]

def encode_char(char):
    encoded_char = str(char.encode("unicode_escape"))
    if "\\u" in encoded_char:
        encoded_char = encoded_char.split("\\u")[1][:-1].upper()
    else:
        if "\\U" in encoded_char:
            encoded_char = get_code_utf16(encoded_char).upper()
    return encoded_char


def run_macbert(config, sorted_df, sep_lst):
  print("Process: Post processing with language model MACBERT")

  cls_lm_weight = float(config.WEIGHT_CORRECT)
  thres = float(config.THRESHOLD)
  ckpt_file = config.MACBERT_CKPT
  folder_path = config.MACBERT_CKPT_FOLDER

  Nom_chars = macbert_correction.intergrate_LM(sorted_df, sep_lst, cls_lm_weight, thres, ckpt_file, folder_path)
  flatten_Nom_chars = []
  for img_idx in range(len(Nom_chars)):
    for sen_idx in range(len(Nom_chars[img_idx])):
      for char_idx in range(len(Nom_chars[img_idx][sen_idx])):
        if Nom_chars[img_idx][sen_idx][char_idx] == 'UKN':
          flatten_Nom_chars.append(Nom_chars[img_idx][sen_idx][char_idx])
        else:
          flatten_Nom_chars.append(encode_char(Nom_chars[img_idx][sen_idx][char_idx]))

  sorted_df["label"] = flatten_Nom_chars
  print("Process done!")

  return sorted_df

def write_final_ocr_results(path, sorted_df, sep_lst):
  print("Process: Write final OCR results to file")

  img_files = sorted_df["source_img"].unique()

  for img_idx, source_img in enumerate(img_files):
    img_df = sorted_df[sorted_df["source_img"] == source_img].reset_index(drop = True)
    source_img = source_img.split(".")[0]
    result_path = os.path.join(path, f"{source_img}.txt")

    with open(result_path, 'w', encoding="utf-16") as f:
        for i in range(0, len(sep_lst[img_idx])):
            if img_df.iloc[i].label == 'UKN':
                f.write("UKN")
                f.write(sep_lst[img_idx][i])    
            else:
                f.write(chr(int(img_df.iloc[i].label, 16)))
                f.write(sep_lst[img_idx][i])

  print("Process done!")
#================================BENCHMARK PHASE================================

def format_gt_txt(df):
    df = df.drop(columns = ["source_img", "img_w", "img_h"])
    df = df[["label", "x1", "y1", "x2", "y2"]]
    return df

def format_dt_txt(df):
    df = df.drop(columns = ["candidate", "score_cls", "source_img"])
    return df  

def save_txt(df, file_name):
    np.savetxt(file_name, df.values, fmt='%s')
    
def create_gt_files(path, GT_DS):
    print("Process: Create groundtruth files")
    
    img_files = GT_DS["source_img"].unique()
    
    with tqdm(total = len(img_files)) as pbar:
        for img in img_files:
            characters = GT_DS[GT_DS["source_img"] == img]
            file_name = img.split(".")[0]
            save_file_name = os.path.join(path, f"{file_name}.txt")

            format_gt = format_gt_txt(characters)
            save_txt(format_gt, save_file_name)
            
            pbar.update()
        
    print("Process done!")
    
def create_dt_files(path, DT_DS):
    print("Process: Create detection files")
    
    img_files = DT_DS["source_img"].unique()
    
    with tqdm(total = len(img_files)) as pbar:
        for img in img_files:
            characters = DT_DS[DT_DS["source_img"] == img]
            file_name = img.split(".")[0]
            save_file_name = os.path.join(path, f"{file_name}.txt")

            format_dt = format_dt_txt(characters)
            save_txt(format_dt, save_file_name)
            
            pbar.update()
        
    print("Process done!")