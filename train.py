import datetime
import time
import os
import pytz

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import sbu_training_root
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir
from model import BDRAR
import torch.nn.functional as F

cudnn.benchmark = True

torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'sbu_hungerstation_glovo'

# batch size of 8 with resolution of 416*416 is exactly OK for the GTX 1080Ti GPU
args = {
    'iter_num': 3000,
    'train_batch_size': 8,
    'last_iter': 0,
    'lr': 5e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

train_set = ImageFolder(sbu_training_root, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
# 1000
tz_paris = pytz.timezone('Europe/Paris')
start_time = datetime.datetime.now(tz_paris).strftime("%Y_%m_%d_%H_%M_%S")
bce_logit = nn.BCEWithLogitsLoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, start_time + '.txt')


def main():
    net = BDRAR().cuda().train()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        train_loss_record, loss_fuse_record, loss1_h2l_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss2_h2l_record, loss3_h2l_record, loss4_h2l_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss1_l2h_record, loss2_l2h_record, loss3_l2h_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss4_l2h_record = AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            fuse_predict, predict1_h2l, predict2_h2l, predict3_h2l, predict4_h2l, \
            predict1_l2h, predict2_l2h, predict3_l2h, predict4_l2h = net(inputs)

            loss_fuse = bce_logit(fuse_predict, labels)
            loss1_h2l = bce_logit(predict1_h2l, labels)
            loss2_h2l = bce_logit(predict2_h2l, labels)
            loss3_h2l = bce_logit(predict3_h2l, labels)
            loss4_h2l = bce_logit(predict4_h2l, labels)
            loss1_l2h = bce_logit(predict1_l2h, labels)
            loss2_l2h = bce_logit(predict2_l2h, labels)
            loss3_l2h = bce_logit(predict3_l2h, labels)
            loss4_l2h = bce_logit(predict4_l2h, labels)

            loss = loss_fuse + loss1_h2l + loss2_h2l + loss3_h2l + loss4_h2l + loss1_l2h + \
                   loss2_l2h + loss3_l2h + loss4_l2h
            loss.backward()

            optimizer.step()

            train_loss_record.update(loss.data, batch_size)
            loss_fuse_record.update(loss_fuse.data, batch_size)
            loss1_h2l_record.update(loss1_h2l.data, batch_size)
            loss2_h2l_record.update(loss2_h2l.data, batch_size)
            loss3_h2l_record.update(loss3_h2l.data, batch_size)
            loss4_h2l_record.update(loss4_h2l.data, batch_size)
            loss1_l2h_record.update(loss1_l2h.data, batch_size)
            loss2_l2h_record.update(loss2_l2h.data, batch_size)
            loss3_l2h_record.update(loss3_l2h.data, batch_size)
            loss4_l2h_record.update(loss4_l2h.data, batch_size)

            curr_iter += 1

            log = '%d, %.5f, %.5f, %.5f, %.5f, ' \
                  '%.5f, %.5f, %.5f, %.5f,  %.5f, ' \
                  '%.5f, %.13f' % \
                  (curr_iter, train_loss_record.avg, loss_fuse_record.avg, loss1_h2l_record.avg, loss2_h2l_record.avg,
                   loss3_h2l_record.avg, loss4_h2l_record.avg, loss1_l2h_record.avg, loss2_l2h_record.avg,
                   loss3_l2h_record.avg, loss4_l2h_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter > args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, '%d.optim_pth' % curr_iter))
                # ---
                # predicted = fuse_predict
                # target = labels
                # fp = ((predicted == 1) & (target == 0)).sum().item()
                # fn = ((predicted == 0) & (target == 1)).sum().item()
                # tp = ((predicted == 1) & (target == 1)).sum().item()
                # tn = ((predicted == 0) & (target == 0)).sum().item()
                #
                # confusion_matrix = torch.tensor([[tp, fp], [fn, tn]])
                #
                # confusion_matrix = confusion_matrix.float() / confusion_matrix.sum()
                # confusion_matrix = confusion_matrix.numpy()
                # print("confusion_matrix", confusion_matrix)

                # from sklearn.metrics import ConfusionMatrixDisplay
                # import matplotlib.pyplot as plt
                # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["masked", "empty"])
                # disp.plot(cmap=plt.cm.Blues)
                # plt.show()
                # ---
                return

# def calculate_matrix(truth, predictions):
#
#     predictions = F.sigmoid(predictions)


# starting_loss = infi
# loss_i < loss_0  then loss_0 = loss_i
# loss_i+1 < loss_0, loss_0 =
# early stopping
if __name__ == '__main__':
    tz_paris = pytz.timezone('Europe/Paris')
    t_s = datetime.datetime.now(tz_paris).strftime("%Y_%m_%d_%H_%M_%S")
    t1 = time.time()
    print("Training start at ".format(t_s))
    main()
    print("training end at {}, system shutdown in 1 minute.".format(datetime.datetime.now(tz_paris).strftime("%Y_%m_%d_%H_%M_%S")))
    print("total training time in hours ", (time.time() - t1) / 3600)
    os.system("sudo shutdown")
