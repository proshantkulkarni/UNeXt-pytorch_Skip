import argparse
import os
from collections import OrderedDict
from glob import glob
import time


import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90,Resize
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool

# from archs_semantic_map import UNext
import archs_CTrans
# import archs

# ARCH_NAMES = archs.__all__
ARCH_NAMES = archs_CTrans.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='isic',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.bmp',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.bmp',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

# args = parser.parse_args()
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        # input = input.cuda()
        # target = target.cuda()
        
        input = input.to('cuda')
        target = target.to('cuda')
        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou,dice = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            # input = input.cuda()
            # target = target.cuda()

            input = input.to('cuda')
            target = target.to('cuda')
            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou,dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main():

    # save_dir = '/content/drive/MyDrive/Amit-Paper3/ISIC_3'
    
    config = vars(parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", device)

    # save_dir = os.path.join("models", config["name"])
    save_dir = os.path.join(os.getcwd(), "models", config["name"])
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n Loading data from: {config['dataset']}")
    print(f" Train images: {os.path.join(config['dataset'], 'train', 'images')}")
    print(f" Train masks:  {os.path.join(config['dataset'], 'train', 'masks')}")
    print(f" Val images:   {os.path.join(config['dataset'], 'val', 'images')}")
    print(f" Val masks:    {os.path.join(config['dataset'], 'val', 'masks')}")
    print(f" Saving results to: {save_dir}\n")

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    # os.makedirs('/content/drive/MyDrive/Amit-Paper3/ISIC_3/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(os.path.join(save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    print(" Saved config to:", os.path.join(save_dir, 'config.yml'))

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        # criterion = nn.BCEWithLogitsLoss().cuda()
        criterion = nn.BCEWithLogitsLoss().to(device)

    else:
        # criterion = losses.__dict__[config['loss']]().cuda()
        criterion = losses.__dict__[config['loss']]().to(device)


    cudnn.benchmark = True

    # create model
    # model = archs.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])

    model = archs_CTrans.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    # model = model.cuda()
    model = model.to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
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
    else:
        raise NotImplementedError


    train_img_paths = glob(os.path.join(config['dataset'], 'train', 'images', '*' + config['img_ext']))
    val_img_paths = glob(os.path.join(config['dataset'], 'val', 'images', '*' + config['img_ext']))

    train_img_ids = []
    for path in train_img_paths:
        img_id = os.path.splitext(os.path.basename(path))[0]
        mask_path = os.path.join(config['dataset'], 'train', 'masks', img_id + config['mask_ext'])
        if os.path.exists(path) and os.path.exists(mask_path):
            train_img_ids.append(img_id)
        if not os.path.exists(path):
            print(f"[WARN] Missing image: {path}")

    val_img_ids = []
    for path in val_img_paths:
        img_id = os.path.splitext(os.path.basename(path))[0]
        mask_path = os.path.join(config['dataset'], 'val', 'masks', img_id + config['mask_ext'])
        if os.path.exists(path) and os.path.exists(mask_path):
            val_img_ids.append(img_id)
        if not os.path.exists(path):
            print(f"[WARN] Missing image: {path}")


    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['dataset'], 'train', 'images'),
        mask_dir=os.path.join(config['dataset'], 'train', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['dataset'], 'val', 'images'),
        mask_dir=os.path.join(config['dataset'], 'val', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=True, persistent_workers=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True, persistent_workers=True)

    print(f" Loaded {len(train_dataset)} training samples")
    print(f" Loaded {len(val_dataset)} validation samples")
    print(f" Total: {len(train_dataset) + len(val_dataset)} samples\n")

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('epoch_time', [])
    ])

    best_iou = 0
    trigger = 0

    total_start = time.time()
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        epoch_start = time.time()
    
        train_log = train(config, train_loader, model, criterion, optimizer)
     
        val_log = validate(config, val_loader, model, criterion)

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        print(f"🕒 Epoch time: {epoch_duration:.2f} seconds")
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['epoch_time'].append(epoch_duration)

        # pd.DataFrame(log).to_csv('/content/drive/MyDrive/Amit-Paper3/ISIC_3/%s/log.csv' %
        #                          config['name'], index=False)

        pd.DataFrame(log).to_csv(os.path.join(save_dir, "log.csv"), index=False)
        
        trigger += 1

        print(f"📈 Epoch [{epoch+1}/{config['epochs']}]:")
        print(f"   Train Loss: {train_log['loss']:.4f}, IOU: {train_log['iou']:.4f}")
        print(f"   Val   Loss: {val_log['loss']:.4f}, IOU: {val_log['iou']:.4f}, Dice: {val_log['dice']:.4f}")

        if val_log['iou'] > best_iou:
            # torch.save(model.state_dict(), '/content/drive/MyDrive/Amit-Paper3/ISIC_3/%s/model.pth' %
            #            config['name'])
            model_path = os.path.join(save_dir, "model.pth")
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
            best_iou = val_log['iou']
            print("=> saved best model")
            print(f" New best model saved at: {model_path}")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

    total_end = time.time()
    total_duration = total_end - total_start
    print(f"\n✅ Training complete in {total_duration / 60:.2f} minutes ({total_duration:.2f} seconds)")

    print("📉 Saving training plots...")

    def plot_and_save(metrics_dict, title, ylabel, save_path):
        plt.figure()
        for label, values in metrics_dict.items():
            plt.plot(values, label=label)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    # 1. Train vs Val Loss
    plot_and_save(
        metrics_dict={
            "Train Loss": log['loss'],
            "Val Loss": log['val_loss']
        },
        title="Loss over Epochs",
        ylabel="Loss",
        save_path=os.path.join(save_dir, 'loss_train_val.png')
    )

    # 2. Train IoU
    plot_and_save(
        metrics_dict={"Train IoU": log['iou']},
        title="Train IoU over Epochs",
        ylabel="IoU",
        save_path=os.path.join(save_dir, 'iou_train.png')
    )

    # 3. Val IoU
    plot_and_save(
        metrics_dict={"Val IoU": log['val_iou']},
        title="Validation IoU over Epochs",
        ylabel="IoU",
        save_path=os.path.join(save_dir, 'iou_val.png')
    )

    # 4. Val Dice
    plot_and_save(
        metrics_dict={"Val Dice": log['val_dice']},
        title="Validation Dice over Epochs",
        ylabel="Dice Score",
        save_path=os.path.join(save_dir, 'dice_val.png')
    )

    print(f"📊 Plots saved to: {save_dir}")


if __name__ == '__main__':
    main()