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

# import archs
# import archs_CTrans
# import archs
# import archs_DCA
# import archs_CTrans_wavelet
# import archs_Fusion
# import archs_wavelet
import archs_DCA_wavelet

from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import Resize
import numpy as np

import sys

class Tee:
    def __init__(self, name, mode="w"):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        # Write to file
        self.file.write(data)
        self.file.flush()
        # For tqdm compatibility, use tqdm.write for non-empty lines without breaking bars
        if data.strip() != "":
            try:
                from tqdm import tqdm
                tqdm.write(data, end='')
            except:
                self.stdout.write(data)
        else:
            self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size override')
    return parser.parse_args()


def main():
    args = parse_args()

    # log_file = open(log_file_path, "w")

    log_file_path = os.path.join("models", args.name, "terminal_output_TEST.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    tee = Tee(log_file_path)

    config_path = os.path.join("models", args.name, "config.yml")
    model_path = os.path.join("models", args.name, "model.pth")

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # with open('/content/drive/MyDrive/Amit-Paper3/ISIC_1_original/isic_exp/config.yml', 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    print('-'*20)
    for key in config.keys():
        print(f'{key}: {config[key]}')
    print('-'*20)


    cudnn.benchmark = True

    print(f"=> creating model {config['arch']}")


    # model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision']).cuda()

    # model = archs_DCA.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision']).cuda()

    # model = archs_CTrans.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision']).cuda()

    # model = archs_CTrans_wavelet.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision']).cuda()

    # model = archs_Fusion.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision']).cuda()

    # model = archs_wavelet.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision']).cuda()

    model = archs_DCA_wavelet.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision']).cuda()

    # model.load_state_dict(torch.load('/content/drive/MyDrive/Amit-Paper3/ISIC_1_original/isic_exp/model.pth', weights_only=True))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Data loading
    test_img_dir = os.path.join(config['dataset'], "test", "images")
    img_ids = glob(os.path.join(test_img_dir, '*' + config['img_ext']))

    # img_ids = glob(os.path.join('/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic/test/images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    val_img_ids = img_ids
    print(len(img_ids))
  

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    test_img_dir = os.path.join(config['dataset'], 'test', 'images')
    test_mask_dir = os.path.join(config['dataset'], 'test', 'masks')

    # val_dataset = Dataset(
    #     img_ids=val_img_ids,
    #     img_dir=os.path.join('/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic/test', 'images'),
    #     mask_dir=os.path.join('/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic/test', 'masks'),
    #     img_ext=config['img_ext'],
    #     mask_ext=config['mask_ext'],
    #     num_classes=config['num_classes'],
    #     transform=val_transform)
    
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=test_img_dir,
        mask_dir=test_mask_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # Metrics storage
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()

    tp_list = [0] * config['num_classes']
    fp_list = [0] * config['num_classes']
    fn_list = [0] * config['num_classes']

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # Compute output
            output = model(input)

            # Calculate metrics
            iou, dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            # Convert output to binary
            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    tp = ((output[i, c] == 1) & (target[i, c].cpu().numpy() == 1)).sum()
                    fp = ((output[i, c] == 1) & (target[i, c].cpu().numpy() == 0)).sum()
                    fn = ((output[i, c] == 0) & (target[i, c].cpu().numpy() == 1)).sum()

                    tp_list[c] += tp
                    fp_list[c] += fp
                    fn_list[c] += fn

                    # Save output images
                    # dest_dir = '/content/drive/MyDrive/Amit-Paper3/results_isic_1_final_original'
                    # dest_dir = os.path.join("results", args.name)
                    dest_dir = os.path.join("models", args.name, "results")
                    os.makedirs(dest_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(dest_dir, f"{meta['img_id'][i]}.png"), (output[i, c] * 255).astype('uint8'))

    print(f'IoU: {iou_avg_meter.avg:.4f}')
    print(f'Dice: {dice_avg_meter.avg:.4f}')

    precision = []
    recall = []
    for c in range(config['num_classes']):
        precision_c = tp_list[c] / (tp_list[c] + fp_list[c] + 1e-9)
        recall_c = tp_list[c] / (tp_list[c] + fn_list[c] + 1e-9)
        precision.append(precision_c)
        recall.append(recall_c)

    overall_precision = np.mean(precision)
    overall_recall = np.mean(recall)

    print("Precision_rrr:", precision)
    print("Recall:", recall)
    print("Overall Precision:", overall_precision)
    print("Overall Recall:", overall_recall)

    # Mean Dice and IoU
    print("Mean Dice:", np.mean([dice_avg_meter.avg]))
    print("Mean IoU:", np.mean([iou_avg_meter.avg]))

    torch.cuda.empty_cache()
    # === Save final metrics ===
    import pandas as pd

# === Save final metrics to CSV ===
    metrics_path = os.path.join("models", args.name, "metrics_results.csv")

    # Create a dictionary of metrics
    metrics = {
        'IoU': [float(iou_avg_meter.avg)],
        'Dice': [float(dice_avg_meter.avg)],
        'Overall_Precision': [float(overall_precision)],
        'Overall_Recall': [float(overall_recall)],
    }

    # Add per-class precision and recall
    for idx, (p, r) in enumerate(zip(precision, recall)):
        metrics[f'Precision_class_{idx}'] = [float(p)]
        metrics[f'Recall_class_{idx}'] = [float(r)]

    # Convert to DataFrame and save
    df = pd.DataFrame(metrics)
    df.to_csv(metrics_path, index=False)

    print(f"✅ Saved metrics to {metrics_path}")

    tee.close()


if __name__ == '__main__':
    main()