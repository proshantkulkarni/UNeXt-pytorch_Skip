import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time

from thop import profile


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args

def compute_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = targets.float()

    TP = (preds * targets).sum()
    TN = ((1 - preds) * (1 - targets)).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()

    epsilon = 1e-7
    sensitivity = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)

    return sensitivity.item(), specificity.item(), accuracy.item()

def main():
    args = parse_args()

    # with open('models/%s/config.yml' % args.name, 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    config_path = os.path.join("models", args.name, "config.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    img_dir = os.path.join(config['dataset'], 'images')
    mask_dir = os.path.join(config['dataset'], 'masks')
    img_ids = glob(os.path.join(img_dir, '*' + config['img_ext']))

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    dummy_input = torch.randn(1, config['input_channels'], config['input_h'], config['input_w']).to('cuda')
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"\n Params: {params / 1e6:.2f} M")
    print(f"  GFLOPs: {macs / 1e9:.2f} G")

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()
    sensitivity_avg_meter = AverageMeter()
    specificity_avg_meter = AverageMeter()
    accuracy_avg_meter = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            

            if count<=5:
                torch.cuda.synchronize()
                start = time.time()

                if config['deep_supervision']:
                    output = model(input)[-1]
                else:
                    output = model(input)

                torch.cuda.synchronize()
                stop = time.time()

                gput.update(stop - start, input.size(0))

                
                start = time.time()
                if config['deep_supervision']:
                    output = model(input)[-1]
                else:
                    output = model(input)
                stop = time.time()

                gput.update(stop-start, input.size(0))

                start = time.time()
                model = model.cpu()
                input = input.cpu()
                output = model(input)
                stop = time.time()

                cput.update(stop-start, input.size(0))
                count=count+1

            iou,dice = iou_score(output, target)
            # output_bin = torch.tensor(output > 0.5).float()
            output_bin = (output > 0.5).float().to(target.device)

            sensitivity, specificity, accuracy = compute_metrics(output_bin, target.cpu())

            sensitivity_avg_meter.update(sensitivity, input.size(0))
            specificity_avg_meter.update(specificity, input.size(0))
            accuracy_avg_meter.update(accuracy, input.size(0))

            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    print('CPU: %.4f' %cput.avg)
    print('GPU: %.4f' %gput.avg)

    print(f"Sensitivity: {sensitivity_avg_meter.avg:.4f}")
    print(f"Specificity: {specificity_avg_meter.avg:.4f}")
    print(f"Accuracy:    {accuracy_avg_meter.avg:.4f}")
    print(f" Avg GPU Time/Image: {gput.avg / config['batch_size']:.4f} sec")
    print(f" Avg CPU Time/Image: {cput.avg / config['batch_size']:.4f} sec")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
