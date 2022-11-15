import os
import random
import time
import argparse


import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler as ampgs
from torch.cuda.amp import autocast as autocast
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations import transforms, geometric
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve

from tensorboardX import SummaryWriter

from scheduler import CosineAnnealingWarmupRestarts
from optimizer import RAdam
from dataset import ClsDataset
from utils import scan_files

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', default=None,
                        help='dataset path')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='batch size')
    config = parser.parse_args()

    return config



def train(train_loader, model, criterion, optimizer, mix_precision):
    model.train()
    train_loss = 0.0
    correct_count = torch.tensor(0, dtype=torch.int16).cuda()
    total_data_num = 0

    start = time.time()    
    for i, training_data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        imgs, labels = training_data
        imgs = imgs.cuda()
        labels = labels.cuda()

        if mix_precision:
            scaler = ampgs()
            with autocast():
                if criterion is not None:
                    output = model(imgs)
                    loss = criterion(output, labels)
                else:
                    loss, output = model(imgs, labels)                

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += scaler.scale(loss).item()

        else:
            if criterion is not None:
                output = model(imgs)
                loss = criterion(output, labels)
            else:
                loss, output = model(imgs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        pred = torch.argmax(output.data, dim=1)
        

        total_data_num += int(pred.size()[0])
        result = labels.int() == pred
        sum_result = torch.sum(result)
        correct_count = correct_count + sum_result
        
    acc = torch.true_divide(correct_count, total_data_num)
    end = time.time()
    print('---Training Loss: %.3f, Training Accuracy: %i/%i=%f,Time:%f' % (
        train_loss/total_data_num, correct_count.data, total_data_num, acc, end - start))
    return train_loss, acc


def val(test_loader, model, criterion, classes_names):
    model.eval()

    with torch.no_grad():
        test_loss = 0.0
        correct_count = 0
        total_data_num = 0

        all_labels = torch.Tensor([]).cuda()
        all_predictions = torch.Tensor([]).cuda()
        all_outputs = torch.Tensor([]).cuda()
        for i, testing_data in enumerate(test_loader, 0):
            imgs, labels= testing_data
            imgs = imgs.cuda()
            labels = labels.cuda()
            
            if criterion is not None:
                output = model(imgs)
                loss = criterion(output, labels)
            else:
                loss, output = model(imgs, labels)
            test_loss += loss.item()
            output = torch.softmax(output.data, dim=1)
            pred = torch.argmax(output, dim=1)
            
            total_data_num += int(pred.size()[0])
            result = labels.int() == pred
            correct_count = correct_count + torch.sum(result).int()
            all_labels = torch.cat((all_labels, labels), dim=0)
            all_predictions = torch.cat((all_predictions, pred), dim=0)
            all_outputs = torch.cat((all_outputs, output), dim=0)

        # sensitivity, specificity = get_spec_by_sens(all_predictions, all_predictions,0.9)
        acc = torch.true_divide(correct_count, total_data_num)

        print(
            '+++Testing Loss: %.3f, Testing Accuracy: %i/%i=%f' % (test_loss, correct_count.data, total_data_num, acc))

        np_all_labels = np.array(all_labels.cpu())
        np_all_prediction = np.array(all_predictions.cpu())


        cm = confusion_matrix(np_all_labels, np_all_prediction)
        cls_report = classification_report(np_all_labels, np_all_prediction, target_names=classes_names)

        # auc = get_auc(all_labels, all_outputs)
        return test_loss, acc, cm, cls_report




def main():
    config = vars(parse_args())
    root_path = config['data_path']
    cv = 4
    n_repeats = 3
    img_ext = '.png'.split(',')
    test_ratio = 0.2
    output_dir = "path/to/output/model/dir"
    input_size = 512
    img_rgb_mean = (0.813, 0.634, 0.733)
    img_rgb_std = (0.091, 0.145, 0.108)
    num_classes = 2
    drop_rate = 0.5
    lr = 0.01
    weights_decay = 1e-4
    min_lr = 0.0001
    first_cycle_steps = 100
    warmup_steps = 5
    cosine_cycle_gamma = 0.9
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    criterion = nn.CrossEntropyLoss()
    train_epoch = 1000
    classes_names = ['0', '1']  # 0:neg, 1:pos
    mix_precision = True
    tf_log_path = 'tf_log'
    exp_name = 'exp_1'
    save_period = 50




    if tf_log_path is not None:
        writer = SummaryWriter(os.path.join(tf_log_path, exp_name),comment=exp_name, flush_secs=1)


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    wsi_file_name_list = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]




    # print(wsi_file_name_list)
    total_wsi = len(wsi_file_name_list)
    random.shuffle(wsi_file_name_list)
    test_num = int(total_wsi*test_ratio)

    train_f_folder = wsi_file_name_list[:-test_num]
    test_f_folder = wsi_file_name_list[-test_num:]

    train_f_list, test_f_list = [], []

    for fd_name in train_f_folder:
        current_fd = os.path.basename(fd_name)
        tmp_files = scan_files(fd_name, img_ext)

        train_f_list.extend([os.path.join(current_fd, f) for f in tmp_files])

    for fd_name in test_f_folder:
        current_fd = os.path.basename(fd_name)
        tmp_files = scan_files(fd_name, img_ext)

        test_f_list.extend([os.path.join(current_fd, f) for f in tmp_files])

    train_label_list = [int(os.path.basename(os.path.dirname(s))) for s in train_f_list]
    test_label_list = [int(os.path.basename(os.path.dirname(s))) for s in test_f_list]

    print('### num of train list:', len(train_label_list))
    print('### num of test list:', len(test_label_list))


    
    train_transform = Compose([
        # geometric.resize.RandomScale(scale_limit=(0.5,1.0), p=1.0),
        geometric.resize.Resize(input_size, input_size),
        geometric.rotate.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
            transforms.RGBShift(),

        ], p=1.),
        OneOf([
            transforms.GaussNoise(),
            transforms.ISONoise(),
            transforms.MultiplicativeNoise(),
            transforms.MedianBlur(),
            transforms.GaussianBlur()
        ], p=1.),
        OneOf([
            transforms.ChannelDropout(),
            transforms.ChannelShuffle(),
            # transforms.CLAHE(),
        ], p=1.),
 
        transforms.Normalize(mean=img_rgb_mean, std=img_rgb_std),
        ToTensorV2()
    ])

    test_transform = Compose([
        geometric.resize.Resize(input_size, input_size),
        transforms.Normalize(mean=img_rgb_mean, std=img_rgb_std),
        ToTensorV2()        
    ])


    test_dataset = ClsDataset(root_path, test_f_list, test_label_list, transform=test_transform)

    pd.DataFrame({'wsi_name': [os.path.basename(f) for f in train_f_folder]}).to_csv(os.path.join(output_dir, 'train_files.csv'), index=False)
    pd.DataFrame({'wsi_name': [os.path.basename(f) for f in test_f_folder]}).to_csv(os.path.join(output_dir, 'test_files.csv'), index=False)


    rkf = RepeatedKFold(n_splits=cv, n_repeats=n_repeats)

    for train_index, val_index in rkf.split(train_f_list):
        current_train_f_list = np.array(train_f_list)[train_index]
        current_train_label_list = np.array(train_label_list)[train_index]
        current_val_f_list = np.array(train_f_list)[val_index]
        current_val_label_list = np.array(train_label_list)[val_index]

        train_dataset = ClsDataset(root_path, current_train_f_list, current_train_label_list, transform=train_transform)
        val_dataset = ClsDataset(root_path, current_val_f_list, current_val_label_list, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False)



        model = timm.create_model("resnest14d", pretrained=False, num_classes=num_classes, drop_rate=drop_rate)

        if torch.cuda.device_count() > 1:
            print('number of GPU > 1, using data parallel')
            model = nn.DataParallel(model)
        model = model.cuda()

        params = filter(lambda p: p.requires_grad, model.parameters())

        optimizer = RAdam(params, lr=lr, weight_decay=weights_decay)

        current_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=first_cycle_steps, max_lr=lr, min_lr=min_lr, 
                                                                 warmup_steps=warmup_steps, gamma=cosine_cycle_gamma)


        best_val_acc = 0
        best_test_acc = 0

        for current_epoch in range(train_epoch):
            print('Epoch [%d/%d]' % (current_epoch, train_epoch))
            print('current lr: {:f}'.format(optimizer.param_groups[0]['lr']))
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, mix_precision)
            val_loss, val_acc, val_cm, val_cls_report = val(val_loader, model, criterion, classes_names)
            test_loss, test_acc, test_cm, test_cls_report = val(test_loader, model, criterion, classes_names)
            current_scheduler.step()
            
            print('#### validation cm: ####')
            print(val_cm)
            print("#### validation report: ####")
            print(val_cls_report)

            print('#### test cm: ####')
            print(test_cm)
            print("#### test report: ####")
            print(test_cls_report)            

            if writer is not None:
                writer.add_scalar('train_loss', train_loss, global_step = current_epoch)
                writer.add_scalar('train_acc', train_acc, global_step = current_epoch)
                writer.add_scalar('val_loss', val_loss, global_step = current_epoch)
                writer.add_scalar('val_acc', val_acc, global_step = current_epoch)
                writer.add_scalar('test_loss', test_loss, global_step = current_epoch)
                writer.add_scalar('test_acc', test_acc, global_step = current_epoch)

            

            if not os.path.isdir(os.path.join(output_dir,'model')):
                os.makedirs(os.path.join(output_dir,'model'))
                
            if val_acc > best_val_acc:
                torch.save(model.module.state_dict(), os.path.join(output_dir,'model','best_val_model.pth'))
                best_val_acc = val_acc
                print("=> saved best val model")
            
            if test_acc > best_test_acc:
                torch.save(model.module.state_dict(), os.path.join(output_dir,'model','test_acc_model.pth'))
                best_test_acc = test_acc
                print("=> saved best test model")

            if (current_epoch+1) % save_period == 0:
                torch.save(model.module.state_dict(), os.path.join(output_dir,'model','{}.pth'.format(str(current_epoch))))


if __name__ == '__main__':
    main()