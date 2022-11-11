import numpy as np
import pandas as pd
import os
import cv2


def check_img_ext(path, img_name):
    possible_ext = ['.jpg', '.png', '.jpeg']
    path = os.path.join(path, img_name)
    for i, ext in enumerate(possible_ext):
        if os.path.exists(path + ext):
            return ext
    print("File extensions are not in the list of \n", possible_ext)

    os.system("ls " + path + "*")


def main(save_fig=False, img_show=False):

    for id_img, img in enumerate(img_names):
        original_img_path = os.path.join(original_path, img)
        original_img = cv2.imread(original_img_path)
        mask_img_path = os.path.join(bdrar_model_dir, img)
        mask_img = cv2.imread(mask_img_path)
        _, thresh_mask = cv2.threshold(mask_img_path, 50, 255, cv2.THRESH_BINARY)

        merge_orig_mask = original_img & cv2.bitwise_not(thresh_mask)
        img_concat = np.concatenate( (original_img, mask_img, merge_orig_mask), axis=1)
        if img_show:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", img_concat)
            key = cv2.waitKey(0)
            if key == ord('n'):
                cv2.destroyAllWindows()


if __name__=='__main__':
    data_dir = '/Users/sky/Documents/data_glovo'
    original_path = os.path.join(data_dir, 'original')
    clipped_path = os.path.join(data_dir, 'clipped')

    bdrar_model_dir = os.path.join(data_dir, 'deep_learning_results', 'BDRAR_glovo_prediction_3001')

    img_names = os.listdir(data_dir)
    main(save_fig=False, img_show=True)
