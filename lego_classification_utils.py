import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import shutil
import random

import tensorflow as tf
import tensorflow.keras as keras


CLASS_NAMES = ['YODA', 'SKY WALKER', 'R2-D2', 'MACE WINDU', 'GENERAL GRIEVIOUS', 
'DARTH VADER', 'RETARD VADER', 'CHICK WITH GUN', 'ZOMBIE BOBBAFETT','DARTH VADER WITHOUT MASK', 'POC']


CLASS_MAP = {f"000{idx}" if idx < 10 else f"00{idx}" : class_name for idx,class_name in enumerate(CLASS_NAMES, 1)}

base_dir = os.path.join(*["lego_classification", "lego_dataset_standard"])


for class_name in CLASS_NAMES:
    os.makedirs(os.path.join(*[base_dir, "train", class_name]), exist_ok=True)
    os.makedirs(os.path.join(*[base_dir, "val", class_name]), exist_ok=True)
    os.makedirs(os.path.join(*[base_dir, "test", class_name]), exist_ok=True)

for class_id,class_name in CLASS_MAP.items():
    total_images = glob.glob(os.path.join(*["lego_classification", "lego_dataset", "star-wars", class_id, "*"]))
    train_samples = int(len(total_images) * 0.6 + 0.5)
    val_samples = int(len(total_images) * 0.25 + 0.5)
    test_samples = len(total_images) - train_samples - val_samples

    train_images = random.choices(total_images, k=train_samples)
    val_images = random.choices(list(set(total_images) - set(train_images)), k=val_samples)
    test_images = list((set(total_images) - set(train_images)) - set(val_images))


    for train_image in train_images:
        shutil.copy(train_image, os.path.join(base_dir, "train", class_name))

    for val_image in val_images:
        shutil.copy(val_image, os.path.join(base_dir, "val", class_name))

    for test_image in test_images:
        shutil.copy(test_image, os.path.join(base_dir, "test", class_name))

    print(f"Class : {class_name}, Total : {len(total_images)}, Train : {train_samples}, Val : {val_samples}, Test : {test_samples}")
