import numpy as np
import os
import cv2

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import sbu_testing_root
from misc import check_mkdir, crf_refine
from model import BDRAR

torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'BDRAR'
args = {
    'snapshot': '3001',
    'scale': 416
}

img_transform = transforms.Compose([
    transforms.Resize(args['scale']),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {'sbu': sbu_testing_root}

to_pil = transforms.ToPILImage()


def calculate_matrix_per_img(ground_truth_path, predictions_path, img_name):
    """
    caluclate the accuracy and balanced error rate (ber)
    :param ground_truth_path:
    :param predictions_path:
    :param img_name:
    :return: ber, accuracy calculated based on Direction-aware spatial context features for shadow detection CVPR 2018.
    """
    if os.path.exists(os.path.join(ground_truth_path, img_name + '.png') ):
        ground_truth_img = os.path.join(ground_truth_path, img_name + '.png')
    elif os.path.exists(os.path.join(ground_truth_path, img_name + '.jpg') ):
        ground_truth_img = os.path.join(ground_truth_path, img_name + '.jpg')
    else:
        print("{} is not png or jpg".format(os.path.join(ground_truth_path, img_name)))
    if os.path.exists(os.path.join(predictions_path, img_name + '.png') ):
        predictions_img = os.path.join(predictions_path, img_name + '.png')
    elif os.path.exists(os.path.join(predictions_path, img_name + '.jpg') ):
        predictions_img = os.path.join(predictions_path, img_name + '.jpg')
    else:
        print("{} is not png or jpg".format(os.path.join(predictions_path, img_name)))

    ground_truth_img = cv2.imread(ground_truth_img, 0)
    predictions_img = cv2.imread(predictions_img, 0)
    _, thresh_prediction = cv2.threshold(predictions_img, 50, 255, cv2.THRESH_BINARY)
    _, thresh_ground_truth = cv2.threshold(ground_truth_img, 50, 255, cv2.THRESH_BINARY)

    t_n = np.sum((thresh_prediction == 0) & (thresh_ground_truth == 0))
    t_p = np.sum((thresh_prediction == 255) & (thresh_ground_truth == 255))
    f_n = np.sum((thresh_prediction == 0) & (thresh_ground_truth == 255))
    f_p = np.sum((thresh_prediction == 255) & (thresh_ground_truth == 0))
    n_p = np.sum(thresh_ground_truth == 255)  # number of shadow pixels
    n_n = np.sum(thresh_ground_truth == 0)  # number of non-shadow pixels
    accuracy = (t_p + t_n) / (n_p + n_n)
    recall = t_p / (t_p + f_n)
    if (n_p == 0 and t_p == 0) or (n_n == 0 and t_n == 0):
        ber = 0
    elif n_p == 0 or n_n == 0:
        ber = None
    else:
        ber = (1 - 0.5*(t_p/n_p + t_n/n_n))*100
    return accuracy, ber


def main():
    net = BDRAR().cuda()

    if len(args['snapshot']) > 0:
        print ('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),map_location='cuda:0'))

    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'ShadowImages')) if
                        img_name.endswith('.jpg') or img_name.endswith('png')]
            # ber_per_img = np.zeros(len(img_list))
            ber_per_img_list = []
            accuracy_per_img_list = []
            for idx, img_name in enumerate(img_list[:]):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                shadow_save_dir = os.path.join(ckpt_path, exp_name, '{}_{}_prediction_{}'.format(exp_name, name, args['snapshot']))
                check_mkdir(shadow_save_dir)
                img = Image.open(os.path.join(root, 'ShadowImages', img_name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                res = net(img_var)  # between 0 and 1, sigmod of each pixel
                prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
                prediction = crf_refine(np.array(img.convert('RGB')), prediction)

                Image.fromarray(prediction).save(os.path.join(shadow_save_dir, img_name))
                accuracy_per_img, ber_per_img = calculate_matrix_per_img(os.path.join(root, 'ShadowMasks'), shadow_save_dir, os.path.splitext(img_name)[0])
                accuracy_per_img_list.append(accuracy_per_img)
                if ber_per_img is not None:
                    ber_per_img_list.append(ber_per_img)
                print("the ber of image {} is {}".format(idx, ber_per_img))
                print("the accuracy of image {} is {}".format(idx, accuracy_per_img))
        average_ber = np.average(ber_per_img_list)
        average_accuracy = np.average(accuracy_per_img_list)
        print("the Balance error rate (BER) and accuracy is ", average_ber, average_accuracy)


if __name__ == '__main__':
    main()
