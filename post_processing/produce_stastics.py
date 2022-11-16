"""
taking the predicted shadow masks whose pixel values are ranging from 0 to 255, the higher the values, the
more confidence for shadow prediction. We put threshold (e.g., 1 to 200 with 20 as one step) for determining if
the pixel is shadow or not. Finally, we can produce a csv with columns as img_name, bi_threshold, ber
each img_name will have N rows corresponding to each threshold and ber
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

import pandas as pd


def calculate_matrix_per_img(ground_truth_path, predictions_path, img_name):
    """
    caluclate the accuracy and balanced error rate (ber)
    :param binary_threshold:
    :param ground_truth_path:
    :param predictions_path:
    :param img_name:
    :return: ber, accuracy calculated based on Direction-aware spatial context features for shadow detection CVPR 2018.
    """
    true_shadow_ext = check_img_ext(ground_truth_path, img_name)
    prediction_shadow_ext = check_img_ext(predictions_path, img_name)

    ground_truth_img = os.path.join(ground_truth_path, img_name + true_shadow_ext)
    predictions_img = os.path.join(predictions_path, img_name + prediction_shadow_ext)
    ground_truth_img = cv2.imread(ground_truth_img, 0)
    predictions_img = cv2.imread(predictions_img, 0)
    kernel = (5,5)
    predictions_img = cv2.GaussianBlur(predictions_img, kernel, 0)
    ground_truth_img = cv2.GaussianBlur(ground_truth_img, kernel, 0)
    h, w = predictions_img.shape

    _, thresh_prediction = cv2.threshold(predictions_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, thresh_ground_truth = cv2.threshold(ground_truth_img, 100, 255, cv2.THRESH_BINARY)  # it should be binary at the beginning

    t_n = np.sum((thresh_prediction == 0) & (thresh_ground_truth == 0))
    t_p = np.sum((thresh_prediction == 255) & (thresh_ground_truth == 255))  # pixel level
    f_n = np.sum((thresh_prediction == 0) & (thresh_ground_truth == 255))
    f_p = np.sum((thresh_prediction == 255) & (thresh_ground_truth == 0))
    n_p = np.sum(thresh_ground_truth == 255)  # number of shadow pixels
    n_n = np.sum(thresh_ground_truth == 0)  # number of non-shadow pixels
    accuracy = (t_p + t_n) / (n_p + n_n)
    shadow_iou = t_p / (np.sum((thresh_prediction == 255)) + np.sum((thresh_ground_truth == 255)) - t_p)
    bkg_iou = t_n / (np.sum((thresh_prediction == 0)) + np.sum((thresh_ground_truth == 0)) - t_n)
    average_iou = 0.5 * (shadow_iou + bkg_iou)
    # TODO think about the situation when denominator == 0
    # recall = t_p / (t_p + f_n)
    # precision =
    if (n_p == 0 and t_p == 0) or (n_n == 0 and t_n == 0):  # never happen in training, but can happen in inference
        ber = 0
    elif n_p == 0 or n_n == 0:   # never happen in training, but can happen in inference
        ber = None
    else:
        ber = (1 - 0.5 * (t_p / n_p + t_n / n_n)) * 100
    print("prediction ", np.amax(predictions_img), np.sum(predictions_img >= 100)/ (h*w), ber)
    return round(accuracy, 3), round(ber, 3), round(average_iou, 3)


def produce_csv():
    img_name_list = []
    thresh_list = []
    ber_list = []
    score_list = []
    for i in pix_scores:
        for i_img, img in enumerate(img_names):
            img_name_list.append(img)
            img_npy = np.load(os.path.join(prediction_shadow_dir, img + '.npy')).flatten()

            thresh_list.append(np.sum(img_npy >= i) / np.sum(img_npy >= 0.01))
            _, ber, _ = calculate_matrix_per_img(truth_shadow_dir, prediction_shadow_dir, img)
            ber_list.append(ber)
            score_list.append(i)

    df = pd.DataFrame(data=zip(img_name_list, thresh_list, score_list , ber_list),
                      columns=['img_name', 'ratio_threshold', 'score_threshold', 'ber'])
    df.to_csv(os.path.join(os.path.dirname(prediction_shadow_dir), 'binary_threshold_ber.csv'))


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_img_ext(path, img_name):
    possible_ext = ['.jpg', '.png', '.jpeg']
    path = os.path.join(path, img_name)
    for i, ext in enumerate(possible_ext):
        if os.path.exists(path + ext):
            return ext
    print("File extensions are not in the list of \n", possible_ext)

    os.system("ls " + path + "*")


if __name__ == '__main__':
    prediction_shadow_dir = '/Users/sky/PycharmProjects/BDRAR_SKY/BDRAR/ckpt/sbu_hungerstation_glovo/2022_11_11_19_07_01/sbu_hungerstation_glovo_hungerstation_prediction_3001'
    truth_shadow_dir = '/Users/sky/PycharmProjects/work_sky/2022_10_26_11_12_00/hungerstation_shadow_dataset/test/shadow_masks'
    img_names = [os.path.splitext(i)[0] for i in os.listdir(truth_shadow_dir)]
    pix_scores = [i for i in np.linspace(0.5, 1.0, 5)]
    produce_csv()


