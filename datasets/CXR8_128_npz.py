import os
import pathlib

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

BASE = os.path.abspath("../..")

#Only use on personal computer
def imgs_to_npz(base_folder, save_folder, filename):
    npz = []


    images = [str(image) for image in pathlib.Path(base_folder).iterdir() if image.is_file()]
    print(len(images))
    output_npz = None
    np_temp = []
    for i, img in enumerate(images):
        #print("Getting images")
        img_arr = cv2.imread(img)
        #print("Converting colors")
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # cv2默认为 bgr 顺序
        resized_img = cv2.resize(img_arr, (128, 128))
        #print("Converting to numpy")
        if output_npz is None:
            output_npz = np.zeros((len(images), resized_img.shape[0], resized_img.shape[1], resized_img.shape[2]))

        output_npz[i, :] = resized_img

        #np_temp.append(img_arr)
        #print("Converting to numpy finished")

        #npz.append(img_arr)
        if i % 1000 == 0:
            print(f"Image {i}/{len(images)}")
            #output_npz = np.array(np_temp) if output_npz is None else np.concatenate((output_npz, np.array(np_temp)))
            #np_temp = []
    print("Loaded all images")

    #output_npz = np.array(npz)
    save_file = os.path.join(save_folder, filename)
    np.savez(save_file, output_npz)
    print(f"{output_npz.shape} size array saved into {save_file}")  # (202599, 64, 64, 3)

if __name__ == '__main__':
    attacks = ["clean",
               "poisoning_simple_replacement-male-No_Finding",
               "poisoning_simple_replacement-Atelectasis-Effusion"]
    base = os.path.join(BASE, "data", "datasets128")
    for attack in attacks:
        data_root = os.path.join(base, attack, "CXR8", "images")

        save_folder = os.path.join(data_root, "..")
        filename = f"CXR8_128_train.npz"
        save_to_file = os.path.join(save_folder, filename)

        if os.path.exists(save_to_file):
            print(f"{save_to_file} already exists, continuing")
            continue
        print(f"Converting {data_root}")
        imgs_to_npz(data_root, save_folder, filename)
