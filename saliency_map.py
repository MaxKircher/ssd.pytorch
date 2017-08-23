import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from layers.modules import MultiBoxLoss

from ssd import build_ssd

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


# From data/VOC0712
VOC_CLASSES = [  # always index 0
'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor']

parser = argparse.ArgumentParser(description='Class Model Visualization for the SSD network')
parser.add_argument('input', help='The image, for which the saliency map shall be computed')
parser.add_argument('--trained_model', default='weights/v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='sal/', type=str,
                    help='File path to save results')
parser.add_argument('--classes', default=VOC_CLASSES, nargs='+', help='The class, that shal be recognised in the image')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

criterion = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False, True)

net = build_ssd('train', 300, 21)    # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
for param in net.parameters():
    param.requires_grad = False
net.eval()

#Get a 300x300 image out of the given image
im = np.swapaxes(cv2.imread(args.save_folder+args.input),0,2).astype('f')
x_off = int((np.size(im, 1)-300)/2)
y_off = int((np.size(im, 2)-300)/2)
im = im[:,x_off:300+x_off,y_off:300+y_off]
input = Variable((torch.from_numpy(np.expand_dims(im, axis=0)).cuda()), requires_grad=True)

# Save the 300x300 image
#im = np.swapaxes(input.data.cpu().numpy()[0],0,2)
#cv2.imwrite(args.save_folder + 'small_input.png', im)

for category in args.classes:
    category_index = VOC_CLASSES.index(category)
    print('New category: ' + category + ' (' + str(category_index) + ')')
    if category_index == -1:
        print('The network isn\'t trained for this category')
        continue

    targets = torch.zeros(1,1,5)
    targets[0,0,2] = 1
    targets[0,0,3] = 1
    targets[0,0,4] = category_index
    targets = Variable(targets, requires_grad=False)

    out = net(input)

    #sim to L1 without location
    loss = -(out[1][0, :, category_index+1].max())

    #5b:
    #loss = (-out[1][0, :, category_index+1]).sort()[0][0:50].mean()


    # L3:
    #loss_l, loss_c = criterion(out, targets)
    #loss = loss_l + loss_c




    loss.backward()
    map = input.grad.data.cpu().numpy()[0]
    map = map.max(0)
    # Normalize, so gradients are visible:
    map = 255*map/map.max()

    vis = np.swapaxes(input.data.cpu().numpy()[0]+map, 0, 2)
    cv2.imwrite(args.save_folder + args.input.split('.')[0] + '_saliency_' + str(category) + '.png', map)
    cv2.imwrite(args.save_folder + args.input.split('.')[0] + '_both_' + str(category) + '.png', vis)

    input.grad.data.zero_()
