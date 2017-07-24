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
parser.add_argument('--trained_model', default='weights/v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='vis/', type=str,
                    help='File path to save results')
parser.add_argument('--lr', '--learning-rate', default=5, type=float, help='initial learning rate')
parser.add_argument('--refine', default='', type=str, help='when set, the given image is refined')
parser.add_argument('--iterations', default=60000, type=int, help='How long the class model shall be optimized')
parser.add_argument('--classes', default=['tvmonitor'], nargs='+', help='The class, that shal be recognised in the image')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
lr = args.lr

torch.set_default_tensor_type('torch.cuda.FloatTensor')


criterion = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False, True)

net = build_ssd('train', 300, 21)    # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
for param in net.parameters():
    param.requires_grad = False
net.eval()

for category in args.classes:
    category_index = VOC_CLASSES.index(category)
    print('New category: ' + category + ' (' + str(category_index) + ')')
    if args.refine == '':
        input = Variable(torch.zeros(1, 3, 300, 300), requires_grad=True)
    else:
        im = np.swapaxes(cv2.imread(args.refine),0,2).astype('f')
        input =Variable((torch.from_numpy(np.expand_dims(im, axis=0)).cuda()), requires_grad=True)

    targets = torch.zeros(1,1,5)
    targets[0,0,2] = 1
    targets[0,0,3] = 1
    targets[0,0,4] = category_index
    targets = Variable(targets, requires_grad=False)

    optimizer = optim.Adam([input], lr=lr)

    for iteration in range(1, args.iterations):
        out = net(input)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        if iteration % 2000 == 0:
            print('loss' + str(loss))
            #im = np.swapaxes(input.data.cpu().numpy()[0],0,2)
            #plt.imshow(im)
            #plt.draw()
        if (iteration % round(args.iterations/4)) == 0:
            lr = lr / 10

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            print('Adjusted learning rate to ' + str(lr))
            print('iteration: ' + str(iteration) + ' loss: ' + str(loss))
        loss.backward()
        optimizer.step()

    im = np.swapaxes(input.data.cpu().numpy()[0],0,2)
    cv2.imwrite(args.save_folder + 'result_' + str(category) + '.png', im)
