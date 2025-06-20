# import argparse
# import os
# from glob import glob

# import cv2
# import torch
# import torch.backends.cudnn as cudnn
# import yaml
# from albumentations.augmentations import transforms
# from albumentations.core.composition import Compose
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm

# import archs
# from dataset import Dataset, InferenceDataset
# from metrics import iou_score, precision_recall
# from utils import AverageMeter
# from albumentations import RandomRotate90,Resize
# import time
# from archs import UNext

# from sklearn.metrics import precision_score, recall_score
# import numpy as np


# def parse_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--name', default=None,
#                         help='model name')

#     args = parser.parse_args()

#     return args


# def main():
#     args = parse_args()

#     with open('/content/drive/MyDrive/Amit-Paper3/results/Monuseg_UNext_woDS/config.yml', 'r') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)

#     print('-'*20)
#     for key in config.keys():
#         print('%s: %s' % (key, str(config[key])))
#     print('-'*20)

#     cudnn.benchmark = True

#     print("=> creating model %s" % config['arch'])
#     model = archs.__dict__[config['arch']](config['num_classes'],
#                                            config['input_channels'],
#                                            config['deep_supervision'])

#     model = model.cuda()

#     # Data loading code
#     img_ids = glob(os.path.join('/content/drive/MyDrive/Amit-Paper3/MoNuSeg/Test_Folder/img', '*' + config['img_ext']))
#     img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

#     # _, val_img_ids = train_test_split(img_ids, test_size=0, random_state=41)

#     val_img_ids = img_ids

#     model.load_state_dict(torch.load('/content/drive/MyDrive/Amit-Paper3/results/Monuseg_UNext_woDS/model.pth',weights_only=True))
#     model.eval()

#     val_transform = Compose([
#         Resize(config['input_h'], config['input_w']),
#         transforms.Normalize(),
#     ])

#     val_dataset = Dataset(
#         img_ids=val_img_ids,
#         img_dir=os.path.join('/content/drive/MyDrive/Amit-Paper3/MoNuSeg/Test_Folder', 'img'),
#         mask_dir=os.path.join('/content/drive/MyDrive/Amit-Paper3/MoNuSeg/Test_Folder', 'labelcol'),
#         img_ext=config['img_ext'],
#         mask_ext=config['mask_ext'],
#         num_classes=config['num_classes'],
#         transform=val_transform)
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=config['num_workers'],
#         drop_last=False)
    
#     # val_dataset = InferenceDataset(
#     #     img_ids=val_img_ids,
#     #     img_dir=os.path.join('/content/drive/MyDrive/MDB-2024/Datasets/MDB_New_Imgs', 'fluorescent_transformed'),
#     #     img_ext=config['img_ext'],
#     #     num_classes=config['num_classes'],
#     #     transform=val_transform)

#     # val_loader = torch.utils.data.DataLoader(
#     #     val_dataset,
#     #     batch_size=1,
#     #     shuffle=False,  # No shuffling for inference
#     #     num_workers=config['num_workers'],
#     #     drop_last=False)

#     iou_avg_meter = AverageMeter()
#     dice_avg_meter = AverageMeter()

#     iou_arr = []
#     dice_arr = []

#     precision_arr = []
#     recall_arr = []

#     tp_list = [0] * config['num_classes']
#     fp_list = [0] * config['num_classes']
#     fn_list = [0] * config['num_classes']
    
#     gput = AverageMeter()
#     cput = AverageMeter()

#     count = 0
#     for c in range(config['num_classes']):
#         os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
#     with torch.no_grad():
#         for input, target, meta in tqdm(val_loader, total=len(val_loader)):
#             # for input, meta in tqdm(val_loader, total=len(val_loader)):
#             input = input.cuda()
#             target = target.cuda()
#             model = model.cuda()
#             # compute output
#             output = model(input)


#             iou,dice = iou_score(output, target)
#             iou_avg_meter.update(iou, input.size(0))
#             dice_avg_meter.update(dice, input.size(0))

#             iou_arr.append(iou)
#             dice_arr.append(dice)

#             output = torch.sigmoid(output).cpu().numpy()
#             output[output>=0.5]=1
#             output[output<0.5]=0

#             for i in range(len(output)):
#                 for c in range(config['num_classes']):
#                     # Compute TP, FP, FN
#                     tp = ((output[i, c] == 1) & (target[i, c].cpu().numpy() == 1)).sum()
#                     fp = ((output[i, c] == 1) & (target[i, c].cpu().numpy() == 0)).sum()
#                     fn = ((output[i, c] == 0) & (target[i, c].cpu().numpy() == 1)).sum()

#                     # Update lists
#                     tp_list[c] += tp
#                     fp_list[c] += fp
#                     fn_list[c] += fn
#                     dest_dir = '/content/drive/MyDrive/Amit-Paper3/results'
#                     os.makedirs(dest_dir, exist_ok=True)
#                     cv2.imwrite(os.path.join(dest_dir,meta['img_id'][i] + '.png'),(output[i, c] * 255).astype('uint8'))

#     print('IoU: %.4f' % iou_avg_meter.avg)
#     print('Dice: %.4f' % dice_avg_meter.avg)

#     precision = []
#     recall = []
#     for c in range(config['num_classes']):
#         precision_c = tp_list[c] / (tp_list[c] + fp_list[c] + 1e-9)  # Add a small value to avoid division by zero
#         recall_c = tp_list[c] / (tp_list[c] + fn_list[c] + 1e-9)  # Add a small value to avoid division by zero
#         precision.append(precision_c)
#         recall.append(recall_c)

#     # Overall precision and recall
#     overall_precision = sum(precision) / len(precision)
#     overall_recall = sum(recall) / len(recall)

#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("Overall Precision:", overall_precision)
#     print("Overall Recall:", overall_recall)

#     print("Dice :", np.mean(dice_arr))
#     print("IoU :", np.mean(iou_arr))

#     # print("Inference Completed")
#     # print("Precision:", np.mean(precision_arr))
#     # print("Recall:", np.mean(recall_arr))

#     torch.cuda.empty_cache()


# if __name__ == '__main__':
#     main()

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
from albumentations import Resize
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    return parser.parse_args()

def main():
    args = parse_args()

    with open('/content/drive/MyDrive/Amit-Paper3/ISIC_1_original/isic_exp/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print(f'{key}: {config[key]}')
    print('-'*20)

    cudnn.benchmark = True

    print(f"=> creating model {config['arch']}")
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision']).cuda()
    model.load_state_dict(torch.load('/content/drive/MyDrive/Amit-Paper3/ISIC_1_original/isic_exp/model.pth', weights_only=True))
    model.eval()

    # Data loading
    img_ids = glob(os.path.join('/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic/test/images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    val_img_ids = img_ids
    print(len(img_ids))
  

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic/test', 'images'),
        mask_dir=os.path.join('/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic/test', 'masks'),
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
                    dest_dir = '/content/drive/MyDrive/Amit-Paper3/results_isic_1_final_original'
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

if __name__ == '__main__':
    main()