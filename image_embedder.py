import os
import glob
import pickle
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image

import models
import datasets


# prepare preprocessing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


def get_model(args):
    # prepare mode
    model = models.__dict__[args.arch](low_dim=128)
    model = torch.nn.DataParallel(model)
    model.cuda()

    assert os.path.isfile(args.checkpoint), f'Model file doesn''t exist -- {args.model}'
    print("=> loading checkpoint '{}'".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    # best_prec1 = checkpoint['best_prec1']
    # lemniscate = checkpoint['lemniscate']
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    return model

def inference_embeddings(model, imgfs):
    entire_features = np.zeros((len(imgfs), 128), np.float32)
    with torch.no_grad():
        for ith, imgf in enumerate(tqdm(imgfs)):
            im = Image.open(imgf)
            x = transform(im)
            features = model(x[None, :].cuda())
            entire_features[ith] = features[0].detach().cpu().numpy()
    return entire_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dir', required=True, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--checkpoint', metavar='PATH')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')

    args = parser.parse_args()

    print(args.dir)
    assert os.path.isdir(args.dir), f'{args.dir} doesn''t exist'


    model = get_model(args)
    model.eval()

    cudnn.benchmark = True

    imgfs = sorted(glob.glob(os.path.join(args.dir, '*')))
    entire_features = inference_embeddings(model, imgfs)
    # entire_features = np.zeros((len(imgfs), 128), np.float32)

    # with torch.no_grad():
    #     for ith, imgf in enumerate(tqdm(imgfs)):
    #         im = Image.open(imgf)
    #         x = transform(im)
    #         features = model(x[None, :].cuda())
    #         entire_features[ith] = features[0].detach().cpu().numpy()

    with open('embedding.pkl', 'wb') as f:
        pickle.dump(
            {'paths': imgfs,
             'features': entire_features}, f)


    # val_dataset = datasets.ImageFolderInstance(args.dir, transform)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # k = 0
    # for x, y, z in tqdm(val_loader):
    #     a = x.shape[0]
    #     print(a)
    #     k+= a
    # print(k)

