import numpy as np
import pandas as pd
import os
import cv2
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from misc import check_mkdir, crf_refine, check_img_ext
from model import BDRAR
import pytz
import datetime


def calculate_matrix_per_img(ground_truth_path, predictions_path, img_name):
    """
    caluclate the accuracy and balanced error rate (ber)
    :param ground_truth_path:
    :param predictions_path:
    :param img_name:
    :return: ber, accuracy calculated based on Direction-aware spatial context features for shadow detection CVPR 2018.
    """
    gti_ext = check_img_ext(ground_truth_path, img_name)
    ground_truth_img = os.path.join(ground_truth_path, img_name + gti_ext)
    pre_ext = check_img_ext(predictions_path, img_name)
    predictions_img = os.path.join(predictions_path, img_name + pre_ext)

    ground_truth_img = cv2.imread(ground_truth_img, 0)
    predictions_img = cv2.imread(predictions_img, 0)
    # _, thresh_prediction = cv2.threshold(predictions_img, 50, 255, cv2.THRESH_BINARY)
    # _, thresh_ground_truth = cv2.threshold(ground_truth_img, 50, 255, cv2.THRESH_BINARY)
    kernel = (5, 5)
    predictions_img = cv2.GaussianBlur(predictions_img, kernel, 0)
    ground_truth_img = cv2.GaussianBlur(ground_truth_img, kernel, 0)

    _, thresh_prediction = cv2.threshold(predictions_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, thresh_ground_truth = cv2.threshold(ground_truth_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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
    return round(accuracy, 3), round(ber, 3), round(average_iou, 3)


def main(calculate_metric=False, save_npy=False):
    net = BDRAR().cuda()
    n_images_infer = 0
    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(
            torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location='cuda:0'))

    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'ShadowImages')) if
                        img_name.endswith('.jpg') or img_name.endswith('png') or img_name.endswith('jpeg')]
            # ber_per_img = np.zeros(len(img_list))
            n_images_infer += len(img_list)
            ber_per_img_list = []
            accuracy_per_img_list = []
            img_name_list = []
            iou_per_img_list = []

            shadow_save_dir = os.path.join(ckpt_path, exp_name, start_time,
                                           '{}_{}_prediction_{}'.format(exp_name, name, args['snapshot']))

            check_mkdir(shadow_save_dir)
            accepted_shadow_dir = os.path.join(shadow_save_dir, 'accepted')
            rejected_shadow_dir = os.path.join(shadow_save_dir, 'rejected')
            check_mkdir(accepted_shadow_dir)
            check_mkdir(rejected_shadow_dir)

            for idx, img_name in enumerate(img_list[:]):
                print('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                img = Image.open(os.path.join(root, 'ShadowImages', img_name)).convert('RGB')
                # TODO add filter from SIPP, too_close, dark_bkg, etc
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                res = net(img_var)  # between 0 and 1, sigmod of each pixel
                img_npy = np.array(res.data.squeeze(0).cpu()).flatten()
                if save_npy:
                    np.save(os.path.join(shadow_save_dir, os.path.splitext(img_name)[0] + '.npy'), img_npy)
                # np.array(res.data.squeeze(0).cpu())  is between 0 and 1

                good_shadow_pix_ratio = np.sum(img_npy >= score_threshold) / np.sum(img_npy >= 0.01)

                prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
                # prediction is between 0 and 255
                prediction = crf_refine(np.array(img.convert('RGB')), prediction)

                if good_shadow_pix_ratio >= 0.0:
                    Image.fromarray(prediction).save(os.path.join(accepted_shadow_dir, img_name))
                    # TODO calculate the number of connected items to filter shadow pixels.
                else:
                    Image.fromarray(prediction).save(os.path.join(rejected_shadow_dir, img_name))

                if calculate_metric:
                    accuracy_per_img, ber_per_img, iou_per_img = calculate_matrix_per_img(os.path.join(root, 'ShadowMasks'),
                                                                             accepted_shadow_dir,
                                                                             os.path.splitext(img_name)[0])
                    accuracy_per_img_list.append(accuracy_per_img)
                    ber_per_img_list.append(ber_per_img)
                    img_name_list.append(os.path.splitext(img_name)[0])
                    iou_per_img_list.append(iou_per_img)
                    print("the ber of image {} is {}".format(idx, ber_per_img))
                    print("the accuracy of image {} is {}".format(idx, accuracy_per_img))

            if calculate_metric:
                average_ber = np.average([i for i in ber_per_img_list if i is not None])
                average_accuracy = np.average(accuracy_per_img_list)
                average_iou = np.average(iou_per_img_list)
                df = pd.DataFrame(data=zip(img_name_list, ber_per_img_list, iou_per_img_list, accuracy_per_img_list),
                                  columns=['img_name', 'ber', 'iou', 'accuracy'])
                csv_name = 'iou_' + str(average_iou) + "_ber_" + str(average_ber) + '.csv'
                df.to_csv(os.path.join(shadow_save_dir, csv_name))
                print("the Balance error rate (BER) and accuracy is ", average_ber, average_iou, average_accuracy)
    return n_images_infer

# def filter_mask():

if __name__ == '__main__':
    # sbu_training_root = './Datasets/SBU_hungerstation_glovo'
    # sbu_testing_root = 'SBU-shadow/SBU-Test/'
    hungerstation_testing_root = '/home/sky/work_sky/shadow_detection/BDRAR/hungerstation_shadow_dataset/test'
    glovo_testing_root = '/home/sky/work_sky/shadow_detection/BDRAR/glovo_shadow_dataset/test'
    yemek_testing_root = '/home/sky/work_sky/shadow_detection/BDRAR/yemek_shadow_dataset/test'
    tz_paris = pytz.timezone('Europe/Paris')
    t_s = datetime.datetime.now(tz_paris).strftime("%Y_%m_%d_%H_%M_%S")

    torch.cuda.set_device(0)

    start_time = datetime.datetime.now(tz_paris).strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_path = './ckpt'
    exp_name = 'sbu_hungerstation_glovo_yemek'
    args = {
        'snapshot': '3501',
        'scale': 416
    }

    img_transform = transforms.Compose([
        transforms.Resize(args['scale']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    to_test = {'glovo': glovo_testing_root,
               'hungerstation': hungerstation_testing_root,
               'yemek': yemek_testing_root
               }
    # to_test = {'yemek': yemek_testing_root}

    to_pil = transforms.ToPILImage()
    ratio_threshold = 0.7
    score_threshold = 0.7

    t1 = time.time()
    print("Training start at ", t_s)
    number_of_inference = main(calculate_metric=True, save_npy=True)
    print("training end at {}, system shutdown in 1 minute.".format(datetime.datetime.now(tz_paris).strftime("%Y_%m_%d_%H_%M_%S")))
    print("Infer {} images takes {} hours ".format(number_of_inference, (time.time() - t1) / 3600))
    print("In average, inference takes {} seconds per image.".format((time.time() - t1) / number_of_inference))
    os.system("sudo shutdown")

