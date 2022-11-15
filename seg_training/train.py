import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler as ampgs
from torch.cuda.amp import autocast as autocast
from torchvision.utils import make_grid

import yaml
from albumentations.augmentations import transforms, geometric
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import cv2

import segmentation_models_pytorch as smp
from dataset import CSVDataset
from metrics import iou_score, iou_score_per_class
from utils import AverageMeter, str2bool
from radam import RAdam

from segmentation_models_pytorch.losses import *
from segmentation_models_pytorch.scheduler import CosineAnnealingWarmupRestarts

ARCH_DICT = {'unetplusplus': smp.UnetPlusPlus,
             'unet': smp.Unet,
             'fpn': smp.FPN,
             'pspnet': smp.PSPNet,
             'pan': smp.PAN,
             'manet': smp.MAnet,
             'linknet': smp.Linknet,
             'deeplabv3': smp.DeepLabV3,
             'deeplabv3plus': smp.DeepLabV3Plus
            }

LOSS_DICT = { 'jaccard': JaccardLoss,
              'dice': DiceLoss,
              'focal': FocalLoss,
              'lovasz': LovaszLoss,
              'soft_bce': SoftBCEWithLogitsLoss,
              'soft_ce': SoftCrossEntropyLoss,
              'tversky': TverskyLoss,
            }


ARCH_NAMES = ARCH_DICT.keys()
LOSS_NAMES = list(LOSS_DICT.keys())
LOSS_NAMES.append('BCEWithLogitsLoss')
MODE_NAMES = ['binary', 'multiclass', 'multilabel']
ACT_NAMES = [None, 'identity',  'sigmoid', 'softmax2d', 'softmax', 'logsoftmax', 'tanh', 'argmax', 'argmax2d']

import albumentations as albu
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
    
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    if preprocessing_fn is not None:
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    else:
         _transform = [
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    return albu.Compose(_transform)


def scan_files(input_file_path, ext_list):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # scan all files and put it into list

        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), "", 1 ))

    return file_list



def add_tb_img(tb_writer, preprocess_params, input_img, ori_img, mask_img, pred_mask, count, prefix_name='img'):
    input_range = preprocess_params['input_range']
    mean = preprocess_params['mean']
    std = preprocess_params['std']

    current_img = input_img.cpu().numpy()

    img_8 = np.ascontiguousarray(current_img)
    img_8[0,:,:] = (img_8[0,:,:]*std[0] + mean[0])*255
    img_8[1,:,:] = (img_8[1,:,:]*std[1] + mean[1])*255
    img_8[2,:,:] = (img_8[2,:,:]*std[2] + mean[2])*255
    img_8[img_8 > 255] = 255
    img_8[img_8 < 0] = 0
    img_8 = img_8.astype(np.uint8)


    img_list = [img_8]

    mask_img = (mask_img * 255).cpu().numpy()

    for mi in mask_img[:]:
        mi = mi.astype(np.uint8)
        
        mi = np.array(cv2.cvtColor(mi,cv2.COLOR_GRAY2RGB)).transpose((2,0,1))

        img_list.append(mi)
    
    pred_mask = pred_mask.detach().cpu().numpy()

    pred_mask = np.where(pred_mask>0.5, 255, 0)


    for mi in pred_mask[:]:
        mi = mi.astype(np.uint8)
        mi = np.array(cv2.cvtColor(mi,cv2.COLOR_GRAY2RGB)).transpose((2,0,1))
        # print('### shape_check 2')
        # print(mi.shape)       
        img_list.append(mi)    
    img_list = np.array(img_list)

        
    final_img = make_grid(torch.as_tensor(img_list))
    tb_writer.add_image('{}_img_list'.format(prefix_name), final_img, count)
    tb_writer.add_image('{}_ori_img'.format(prefix_name), ori_img, count)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='unetplusplus',
                        choices=ARCH_NAMES,
                        help='architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: unetplusplus)')
    parser.add_argument('--mode', default='multilabel',
                        choices=MODE_NAMES,
                        help='model: ' +
                        ' | '.join(MODE_NAMES) +
                        ' (default: multilabel)')
    parser.add_argument('--act', default=None,
                        help='activation: ' +
                        ' | '.join(ACT_NAMES[1:]) +
                        ' (default: identity)')
    parser.add_argument('--encoder', default='resnet18',
                        help='encoder name, deafult is resnet18')
    parser.add_argument('--encoder_weight', default=None,
                        help='encoder weight, default is None')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--mix_precision', default=False, type=str2bool)
    parser.add_argument('--show_arch', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='tversky',
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEWithLogitsLoss)\n'+
                        'multiple loss supported: using , to split, e.g. BCEWithLogitsLoss,lovasz')
    parser.add_argument('--loss_weight', default='1.',
                        help='loss weight: ' +
                        'multiple loss weight supported: using , to split, e.g. 0.5,1.')
    # dataset
    parser.add_argument('--dataset_root', default='input',
                        help='dataset root path')
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'RAdam'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD', 'RAdam']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR', 'CosineAnnealingWarmupRestarts'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--dp_rate', default=-1., type=float)
    parser.add_argument('--class_name', default='nlst,cn',
                        help='class name list')

    parser.add_argument('--tf_log_path', default=None, type=str)
    
    parser.add_argument('--cosine_cycle_steps', default=20, type=int)
    parser.add_argument('--cosine_cycle_warmup', default=2, type=int)
    parser.add_argument('--cosine_cycle_gamma', default=0.9, type=float)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer, preprocess_params=None, tb_writer=None):


    per_cls_avg_meter = [AverageMeter() for _ in range(config['num_classes'])]
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    count = 0
    for input, target, _, ori_img in train_loader:

        input = input.cuda()

        target = target.cuda()

        # compute output
        if config['mix_precision']:
            scaler = ampgs()

            with autocast():
                if config['deep_supervision']:
                    outputs = model(input)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, target)
                    loss /= len(outputs)
                    iou, iou_per_cls = iou_score_per_class(outputs[-1], target)
                    output = outputs[-1]
                else:
                    output = model(input)

                    loss = criterion(output, target)
                    iou, iou_per_cls = iou_score_per_class(output, target)
            # compute gradient and do optimizing step
            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, iou_per_cls = iou_score_per_class(outputs[-1], target)
                output = outputs[-1]
            else:
                output = model(input)

                loss = criterion(output, target)
                iou, iou_per_cls = iou_score_per_class(output, target)

            # compute gradient and do optimizing step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        for tmp_m, tmp_v in zip(per_cls_avg_meter, iou_per_cls):
            tmp_m.update(tmp_v, input.size(0))

        possfix_list = [('loss', avg_meters['loss'].avg),('iou', avg_meters['iou'].avg),]

        for i, cls_name in enumerate(config['class_name']):
            possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))


        if preprocess_params is not None and tb_writer is not None:
            add_tb_img(tb_writer, preprocess_params, input[0], ori_img[0], target[0], output[0], count, prefix_name='train')
            count+=1


        postfix = OrderedDict(possfix_list)
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    possfix_list = [('loss', avg_meters['loss'].avg),('iou', avg_meters['iou'].avg),]
    for i, cls_name in enumerate(config['class_name']):
        possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))

    return OrderedDict(possfix_list)


def validate(config, val_loader, model, criterion, preprocess_params=None, tb_writer=None):
    per_cls_avg_meter = [AverageMeter() for _ in range(config['num_classes'])]
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        count = 0
        for input, target, _, ori_img in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, iou_per_cls = iou_score_per_class(outputs[-1], target)
                output = outputs[-1]
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, iou_per_cls = iou_score_per_class(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            for tmp_m, tmp_v in zip(per_cls_avg_meter, iou_per_cls):
                tmp_m.update(tmp_v, input.size(0))

            possfix_list = [('loss', avg_meters['loss'].avg),('iou', avg_meters['iou'].avg),]

            for i, cls_name in enumerate(config['class_name']):
                possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))

            if preprocess_params is not None and tb_writer is not None:
                add_tb_img(tb_writer, preprocess_params, input[0], ori_img[0], target[0], output[0], count, prefix_name='val')
                count+=1

            postfix = OrderedDict(possfix_list)

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    possfix_list = [('loss', avg_meters['loss'].avg),('iou', avg_meters['iou'].avg),]
    for i, cls_name in enumerate(config['class_name']):
        possfix_list.append(('iou_{}'.format(cls_name), per_cls_avg_meter[i].avg))

    return OrderedDict(possfix_list)



def main():
    config = vars(parse_args())

    if config['act'] == None:
        if config['mode'] == 'multiclass':
            config['act'] = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
        else:
            config['act'] = None

    config['class_name'] = config['class_name'].rstrip().split(',')
    if len(config['class_name']) != config['num_classes']:
        raise Exception('num of class name not equal to num_classes in config file.')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    tf_log_path = config['tf_log_path']
    tf_save_name = '{}_{}_{}_{}_{}'.format(
                                           config['name'], config['arch'], config['loss'],
                                           config['optimizer'], str(config['lr'])
                                          )

    writer = None
    if tf_log_path is not None:
        writer = SummaryWriter(os.path.join(tf_log_path, tf_save_name),comment=tf_save_name, flush_secs=1)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)


    if writer is not None:
        config_content = ''
        for k in sorted(config.keys()):
            config_content += '{}: {}\n\n'.format(k, str(config[k]))
        writer.add_text('config:', config_content)


    # define loss function (criterion)
    config['loss'] = config['loss'].rstrip().split(',')
    config['loss_weight']= config['loss_weight'].rstrip().split(',')

    assert len(config['loss']) == len(config['loss_weight']), \
           'num of loss must equal to num of loss weight, {} vs {}'.format(str(len(config['loss'])), str(len(config['loss_weight'])))

    loss_list = []
    loss_w_list = []
    for loss_name, loss_w in zip(config['loss'], config['loss_weight']):
        if loss_name == 'BCEWithLogitsLoss':
            current_criterion = nn.BCEWithLogitsLoss()
        else:
            current_criterion = LOSS_DICT[loss_name](mode=config['mode'])
        loss_list.append(current_criterion)
        loss_w_list.append(float(loss_w))

    criterion = JointLoss(loss_list, loss_w_list)

    criterion = criterion.cuda()
    cudnn.benchmark = True

####################################################################################
    # create model
    # create segmentation model with pretrained encoder
    model = ARCH_DICT[config['arch']](
        encoder_name=config['encoder'],
        encoder_weights=config['encoder_weight'],
        classes=config['num_classes'],
        activation=config['act'],
    )

    if config['encoder_weight'] is not None:
        preprocessing_fn, preprocess_params = smp.encoders.get_preprocessing_fn(config['encoder'], config['encoder_weight'])
    else:
        preprocessing_fn = None
    
    print('### debug encoder weight params')
    print(preprocess_params)

##############################################################################
    if torch.cuda.device_count() > 1:
        print('number of GPU > 1, using data parallel')
        model = nn.DataParallel(model)

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'RAdam':
         optimizer = RAdam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    elif config['scheduler'] == 'CosineAnnealingWarmupRestarts':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=config['cosine_cycle_steps'], max_lr=config['lr'], min_lr=config['min_lr'],
                                                                warmup_steps=config['cosine_cycle_warmup'], gamma=config['cosine_cycle_gamma'])
    else:
        raise NotImplementedError

    # Data loading
    train_transform = Compose([
        geometric.rotate.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(hue_shift_limit=10),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
            transforms.RGBShift(),

        ], p=1.),
        OneOf([
            transforms.GaussNoise(),
            transforms.ISONoise(),
            transforms.MultiplicativeNoise(),
            transforms.MedianBlur(blur_limit=3),
            transforms.GaussianBlur()
        ], p=1.),
        OneOf([
            transforms.ChannelDropout(),
            transforms.ChannelShuffle(),
            transforms.ToGray(),
            transforms.ToSepia(),
        ], p=1.),
        transforms.CoarseDropout(min_holes=4, min_height=4, min_width=4, max_holes=20),
        geometric.resize.Resize(config['input_h'], config['input_w']),

    ])

    val_transform = Compose([
        geometric.resize.Resize(config['input_h'], config['input_w']),

    ])

    train_dataset = CSVDataset(
        img_dir=os.path.join(config['dataset_root'], config['dataset'] + '_train', 'images'),
        mask_dir=os.path.join(config['dataset_root'], config['dataset'] + '_train', 'masks'),
        class_name_list=config['class_name'],
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        target_patch_num=int(config['batch_size']*200),
        preprocessing=get_preprocessing(preprocessing_fn),
        transform=train_transform,
        mode=config['mode'],
        cust_trans=True)
    val_dataset = CSVDataset(
        img_dir=os.path.join(config['dataset_root'], config['dataset'] + '_test', 'images'),
        mask_dir=os.path.join(config['dataset_root'], config['dataset'] + '_test', 'masks'),
        class_name_list=config['class_name'],
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        target_image_size=[720, 720],
        target_patch_num=int(config['batch_size']*100),
        preprocessing=get_preprocessing(preprocessing_fn),
        transform=val_transform,
        area_thresh=0.1,
        mode=config['mode'])


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)



    # log init
    log_list = [
                ('epoch', []),
                ('lr', []),
                ('loss', []),
                ('iou', []),
                ('val_loss', []),
                ('val_iou', []),
               ]

    for i, cls_name in enumerate(config['class_name']):
        log_list.append(('iou_{}'.format(cls_name), []))
    for i, cls_name in enumerate(config['class_name']):
        log_list.append(('val_iou_{}'.format(cls_name), []))

    log = OrderedDict(log_list)

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        print('current lr: {:f}'.format(optimizer.param_groups[0]['lr']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer, preprocess_params, writer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion, preprocess_params, writer)

        if config['scheduler'] == 'CosineAnnealingLR' or config['scheduler'] == "CosineAnnealingWarmupRestarts":
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        train_msg = ''
        val_msg = ''
        for i, cls_name in enumerate(config['class_name']):
            train_msg += 'train_iou_{} {:04f} - '.format(cls_name, train_log['iou_{}'.format(cls_name)])
        train_msg = train_msg[:-2]
        for i, cls_name in enumerate(config['class_name']):
            val_msg += 'val_iou_{} {:04f} - '.format(cls_name, val_log['iou_{}'.format(cls_name)])
        val_msg = val_msg[:-2]
        print(train_msg)
        print(val_msg)


        if writer is not None:
            writer.add_scalar('Train_Loss', train_log['loss'], global_step=epoch)
            writer.add_scalar('Val_Loss', val_log['loss'], global_step=epoch)
            writer.add_scalar('Train_IoU', train_log['iou'], global_step=epoch)
            writer.add_scalar('Val_IoU', val_log['iou'], global_step=epoch)
            for i, cls_name in enumerate(config['class_name']):
                writer.add_scalar('Train_{}_IoU'.format(cls_name), train_log['iou_{}'.format(cls_name)], global_step=epoch)
            for i, cls_name in enumerate(config['class_name']):
                writer.add_scalar('Val_{}_IoU'.format(cls_name), val_log['iou_{}'.format(cls_name)], global_step=epoch)

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        for i, cls_name in enumerate(config['class_name']):
            log['iou_{}'.format(cls_name)].append(train_log['iou_{}'.format(cls_name)])

        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        for i, cls_name in enumerate(config['class_name']):
            log['val_iou_{}'.format(cls_name)].append(val_log['iou_{}'.format(cls_name)])


        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
