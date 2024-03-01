import argparse
import os
from glob import glob
from torch import nn
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
from archs import UNext

import torch_pruning as tp

import torchvision

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="BUSI_UNext_woDS",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # config['img_ext'] = ".jpg"

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'training', 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=0)

    val_img_ids = img_ids


    model.load_state_dict(torch.load('models/%s/model.pth'%
                                     config['name']))

    # print(model)
    #
    # config_list = [{
    #     'sparsity_per_layer': 0.2,
    #     'op_types': ['Conv2d']
    # }, {
    #     'exclude': True,
    #     'op_names': ['final']
    # }]
    #
    # from nni.compression.pytorch.pruning import L1NormPruner
    # pruner = L1NormPruner(model, config_list)
    # #
    # # compress the model and generate the masks
    # _, masks = pruner.compress()
    # # show the masks sparsity
    # for name, mask in masks.items():
    #     print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))
    
    # # need to unwrap the model, if the model is wrapped before speedup
    # pruner._unwrap_model()
    
    # # speedup the model
    # from nni.compression.pytorch.speedup import ModelSpeedup
    
    # ModelSpeedup(model, torch.rand(1, 3, 256, 256).cuda(), masks).speedup_model()
    
    # print(model)


    # example_inputs = torch.randn(8, 3, 256, 256).cuda()

    # # 1. 选择合适的重要性评估指标，这里使用权值大小
    # # imp = tp.importance.MagnitudeImportance(p=2)
    # imp = tp.importance.BNScaleImportance()


    # # 2. 忽略无需剪枝的层，例如最后的分类层（总不能剪完类别都变少了叭？）
    # ignored_layers = []
    # for m in model.modules():
    #     if isinstance(m, torch.nn.Conv2d) and m.out_channels == 1:
    #         ignored_layers.append(m)  # DO NOT prune the final classifier!

    # # 3. 初始化剪枝器
    # iterative_steps = 5  # 迭代式剪枝，重复5次Pruning-Finetuning的循环完成剪枝。
    # pruner = tp.pruner.MagnitudePruner(
    #     model,
    #     example_inputs,  # 用于分析依赖的伪输入
    #     importance=imp,  # 重要性评估指标
    #     iterative_steps=iterative_steps,  # 迭代剪枝，设为1则一次性完成剪枝
    #     ch_sparsity=0.1,  # 目标稀疏性，这里我们移除50%的通道 ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    #     ignored_layers=ignored_layers,  # 忽略掉最后的分类层
    # )

    # # 4. Pruning-Finetuning的循环
    # base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # for i in range(iterative_steps):
    #     pruner.step()
    #     macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)


    # print(model)



    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'training', 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'training', 'masks'),
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

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)


            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
