import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from random import sample


def main(save_img=False):
    df = pd.read_csv(csv_path)

    df_best_shadow = df[df.best_shadow == 1]
    img_name_best_shadow = df_best_shadow['img_name'].to_list()

    # get image name and remove dublicates
    img_name_list = []
    for i_img, img in enumerate(img_name_best_shadow):
        img_name_list.append(os.path.splitext(img)[0])

    img_name_no_duplicates = list(dict.fromkeys(img_name_list))

    print("img nb before and after removing duplicates {} {}".format(len(img_name_list)
                                                                     , len(img_name_no_duplicates)))
    print("number of shadow images ", len(img_name_no_duplicates))
    n_shadow = 0
    for i, img in enumerate(img_name_no_duplicates):
        original_img = os.path.join(original_path, img + '.jpeg')
        clip_img = os.path.join(clipped_path, img + '.jpeg' )
        shadow_msk = os.path.join(full_shadow_mask_path, img + '.jpg')
        gray_mask = cv2.imread(shadow_msk, 0)

        if os.path.exists(shadow_msk):
            print("processing {}".format(i))
            n_shadow += 1
            os.system('cp ' + original_img + ' ' + shadow_img_path + '/')
            os.system('cp ' + clip_img + ' ' + obj_msk_path + '/')
            # os.system('cp ' + shadow_msk + ' ' + shadow_msk_path + '/')
            cv2.imwrite(os.path.join(shadow_msk_path, img + '.jpg'), gray_mask)
            if save_img:
                fig, axes = plt.subplots(1, 3)
                ax = axes.ravel()
                [axi.set_axis_off() for axi in ax.ravel()]
                img_1 = cv2.imread(original_img)
                img_2 = cv2.imread(clip_img)
                img_3 = cv2.imread(shadow_msk)

                ax[0].imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
                ax[0].set_title("Original")
                ax[1].imshow(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
                ax[1].set_title("clipped object")
                ax[2].imshow(cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY))
                ax[2].set_title("shadow mask")
                plt.savefig(os.path.join(test_img_save_dir, img + '.jpg'))
                # Clear the current axes.
                plt.cla()
                # Clear the current figure.
                plt.clf()
                # Closes all the figure windows.
                plt.close('all')

    print("number of images in the final dataset", n_shadow)


def check_img_ext(path, img_name):
    possible_ext = ['.jpg', '.png', '.jpeg']
    path = os.path.join(path, img_name)
    for i, ext in enumerate(possible_ext):
        if os.path.exists(path + ext):
            return ext
    print("File extensions are not in the list of \n", possible_ext)

    os.system("ls " + path + "*")
    return None


def select_training_test_validation(data_path):
    dataset_names = ['training', 'test', 'validation']
    dir_names = ['shadow_images', 'shadow_masks']
    img_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(data_path, 'shadow_images'))]
    img_names.sort()
    img_validation = sample(img_names, 10)
    img_train_test = list(set(img_names) - set(img_validation))
    n_img_train = int(len(img_train_test) * 0.8)
    img_train = img_train_test[:n_img_train]
    img_test = img_train_test[n_img_train:]
    n_img_list = [img_train, img_test, img_validation]
    img_names_per_set = {}
    for i, set_name in enumerate(dataset_names):
        img_names_per_set[set_name] = n_img_list[i]

    for i in dir_names:

        for j in dataset_names:
            dataset_path = os.path.join(data_path, j, i)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            for img in img_names_per_set[j]:
                ext = check_img_ext(os.path.join(data_path, i), img)
                source = os.path.join(data_path, i, img + ext)
                destination = os.path.join(dataset_path, img + ext)
                os.system('cp ' + source + ' ' + destination)


if __name__ == '__main__':
    csv_name = 'select_best_shadow.csv'
    data_dir = '/Users/sky/Documents/data_glovo/select_shadow_after_inference2022_11_10_09_39_27'
    csv_path = os.path.join(data_dir, csv_name)
    full_shadow_mask_path = os.path.join(data_dir, 'shadow_mask_RGB')  # merged shadow mask without selection
    # original_path = "/Users/sky/Documents/s3_data/round1/7K_sample"
    # clipped_path = "/Users/sky/Documents/s3_data/round1/7K_clipped"
    original_path = "/Users/sky/Documents/data_glovo/original"
    clipped_path = "/Users/sky/Documents/data_glovo/clipped"

    shadow_dataset_name = 'glovo_shadow_dataset'
    shadow_img_path = os.path.join(data_dir, shadow_dataset_name, 'shadow_images')  # original img after selection
    shadow_msk_path = os.path.join(data_dir, shadow_dataset_name, 'shadow_masks')  # shadow mask after selection
    obj_msk_path = os.path.join(data_dir, shadow_dataset_name, 'object_masks')  # object mask after selection
    test_img_save_dir = os.path.join(data_dir, shadow_dataset_name, 'debug_img')

    if not os.path.exists(shadow_img_path):
        os.makedirs(shadow_img_path)
    if not os.path.exists(shadow_msk_path):
        os.makedirs(shadow_msk_path)
    if not os.path.exists(obj_msk_path):
        os.makedirs(obj_msk_path)
    if not os.path.exists(test_img_save_dir):
        os.makedirs(test_img_save_dir)
    # main(save_img=False)

    select_training_test_validation(os.path.join(data_dir, shadow_dataset_name))
