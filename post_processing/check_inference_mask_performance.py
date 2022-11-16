import sys

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt


def check_img_ext(path, img_name):
    possible_ext = ['.jpg', '.png', '.jpeg']
    path = os.path.join(path, img_name)
    for i, ext in enumerate(possible_ext):
        if os.path.exists(path + ext):
            return ext
    print("File extensions are not in the list of \n", possible_ext)

    os.system("ls " + path + "*")


def mask_bgr2binary(mask_img):
    mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(mask_gray, (5, 5), 0)
    _, thresh_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_mask


def main(save_fig=False, img_show=False):

    img_name_list = []
    decision_list = []
    for id_img, img in enumerate(img_names[:]):
        real_id = id_img
        print("analyzing image {} with name {}".format(real_id, img))
        img_name_list.append(img)
        orig_ext = check_img_ext(original_path, img)
        original_img_path = os.path.join(original_path, img + orig_ext)
        original_img = cv2.imread(original_img_path)

        msk_ext = check_img_ext(bdrar_model_dir, img)
        mask_img_path = os.path.join(bdrar_model_dir, img+msk_ext)
        mask_img = cv2.imread(mask_img_path)
        thresh_mask = mask_bgr2binary(mask_img)
        mask_img_path = os.path.join(first_results_dir, img+msk_ext)
        mask_img_0 = cv2.imread(mask_img_path)
        thresh_mask_0 = mask_bgr2binary(mask_img_0)

        merge_orig_mask = cv2.bitwise_and(original_img, original_img, mask=cv2.bitwise_not(thresh_mask))
        merge_orig_mask_0 = cv2.bitwise_and(original_img, original_img, mask=cv2.bitwise_not(thresh_mask_0))

        img_concat = np.concatenate((original_img, mask_img, merge_orig_mask), axis=1)
        if img_show:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", img_concat)
            key = cv2.waitKey(0)
            if key == ord('s'):
                decision_list.append(1)
                cv2.destroyAllWindows()
            elif key == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()
            else:
                decision_list.append(0)
                cv2.destroyAllWindows()

        if save_fig:
            fig, axes = plt.subplots(2, 3)
            ax = axes.ravel()
            [axi.set_axis_off() for axi in ax.ravel()]
            ax[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
            ax[0].set_title("Original")
            ax[1].imshow(cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY))
            ax[1].set_title("mask_second_itereation")
            ax[2].imshow(cv2.cvtColor(merge_orig_mask, cv2.COLOR_BGR2RGB))
            ax[2].set_title("merged_second_itereation")
            ax[3].imshow(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
            ax[3].set_title("Original")
            ax[4].imshow(cv2.cvtColor(mask_img_0, cv2.COLOR_BGR2GRAY))
            ax[4].set_title("mask_first_itereation")
            ax[5].imshow(cv2.cvtColor(merge_orig_mask_0, cv2.COLOR_BGR2RGB))
            ax[5].set_title("merged_first_itereation")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, img + '.jpg'))
            plt.show()
            plt.cla()
            plt.clf()
            plt.close('all')


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

if __name__ == '__main__':
    # original_dir = "/Users/sky/Documents/data_glovo/original"
    # clipped_dir = "/Users/sky/Documents/data_glovo/clipped"
    original_path = "/Users/sky/Documents/s3_data/round1/7K_sample"
    clipped_path = "/Users/sky/Documents/s3_data/round1/7K_clipped"
    # bdrar_model_dir = '/Users/sky/PycharmProjects/BDRAR_SKY/BDRAR/ckpt/sbu_hungerstation_glovo/2022_11_16_15_04_00/sbu_hungerstation_glovo_hungerstation_prediction_3501/accepted'
    # first_results_dir = '/Users/sky/PycharmProjects/BDRAR_SKY/BDRAR/ckpt/BDRAR/calculate_precision_recall/BDRAR_hungerstation_prediction_3001_with_npy'
    first_results_dir = '/Users/sky/PycharmProjects/BDRAR_SKY/BDRAR/ckpt/sbu_hungerstation_glovo/2022_11_16_15_04_00/sbu_hungerstation_glovo_hungerstation_prediction_3501/accepted'
    bdrar_model_dir = '/Users/sky/PycharmProjects/BDRAR_SKY/BDRAR/ckpt/sbu_hungerstation_glovo_yemek/2022_11_16_14_39_51/sbu_hungerstation_glovo_yemek_hungerstation_prediction_3501/accepted'

    img_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(bdrar_model_dir)
        if os.path.isfile(os.path.join(bdrar_model_dir, f))
    ]
    results_dir = os.path.join(os.path.dirname(bdrar_model_dir), 'performance_check')
    check_mkdir(results_dir)

    main(save_fig=True, img_show=False)