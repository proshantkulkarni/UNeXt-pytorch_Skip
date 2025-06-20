import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def precision_recall(predictions, targets):
    predictions = predictions.cpu().numpy() 
    targets = targets.cpu().numpy() 

    print(np.unique(predictions, return_counts=True), targets)

    true_positives = ((predictions == 1) & (targets == 1)).sum()
    false_positives = ((predictions == 1) & (targets == 0)).sum()
    false_negatives = ((predictions == 0) & (targets == 1)).sum()

    # print(true_positives, false_positives, false_negatives)

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    return precision, recall

def recall(predictions, targets):
    true_positives = sum((p == 1 and t == 1) for p, t in zip(predictions, targets))
    false_negatives = sum((p == 0 and t == 1) for p, t in zip(predictions, targets))
    
    if true_positives + false_negatives == 0:
        return 0
    
    return true_positives / (true_positives + false_negatives)