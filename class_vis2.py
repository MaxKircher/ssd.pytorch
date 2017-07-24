import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from layers.modules import MultiBoxLoss

from data import VOC_CLASSES as labelmap

from ssd import build_ssd
from eval import voc_eval

import os
import argparse
import numpy as np
import cv2
import pickle


parser = argparse.ArgumentParser(description='Class Model Visualization for the SSD network')
parser.add_argument('--trained_model', default='weights/v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='vis/', type=str,
                    help='File path to save results')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--iterations', default=50000, type=int, help='How long the class model shall be optimized')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

criterion = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False, True)

net = build_ssd('test', 300, 21)    # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
#net.train() Useless?

#input = Variable(torch.rand(1, 3, 300, 300)*255, requires_grad=True)
input = Variable(torch.zeros(1, 3, 300, 300)*255, requires_grad=True)
im = np.swapaxes(input.data.cpu().numpy()[0],0,2)
cv2.imwrite(args.save_folder + 'random_new.png', im)

#targets = Variable(torch.FloatTensor([[[0, 0, 300, 300, 1]]]), requires_grad=True)
targets = torch.zeros(1,1,5)
targets[0,0,2] = 1
targets[0,0,3] = 1
targets[0,0,4] = 5
targets = Variable(targets, requires_grad=False)

#optimizer = optim.SGD([input], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = optim.Adam([input], lr=args.lr)

for iteration in range(0, args.iterations):
    out = net(input).data

    optimizer.zero_grad()

    # Copied from eval.test_net:

    num_images = 1
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # skip j = 0, because it's the background class
    for j in range(1, out.size(1)):
        dets = out[0, j, :]
        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        if dets.dim() == 0:
            continue
        boxes = dets[:, 1:]
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h
        scores = dets[:, 0].cpu().numpy()
        cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        all_boxes[j][i] = cls_dets

    dog_det_file = 'vis/Annotations/dog_det.pkl'
    with open(dog_det_file, 'wb') as f:
        pickle.dump(all_boxes, f)

    det_text = 'vis/Annotations/dog_det.pkl'
    with open(det_text, 'wt') as f:
        dets = all_boxes[0+1][0]
        if dets != []:
            # the VOCdevkit expects 1-based indices
            for k in range(dets.shape[0]):
                f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                        format(index[1], dets[k, -1],
                            dets[k, 0] + 1, dets[k, 1] + 1,
                            dets[k, 2] + 1, dets[k, 3] + 1))



    anno_path = 'vis/Annotations/%s.xml'
    image_set_file = 'vis/Annotations/images'
    classname = 'dog'
    cachedir = 'vis/Annotations/'
    _, _, ap = voc_eval(det_text, anno_path, image_set_file.format('test'), classname, cachedir)

    if iteration % 10 == 0:
        print('loss' + str(ap))

    ap.backward()
    optimizer.step()






    '''
    # backprop
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, targets)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    '''
    #im = np.swapaxes(input.data.cpu().numpy()[0],0,2)
    #cv2.imwrite(args.save_folder + 'evolution' + str(iteration) + '.png', im)

im = np.swapaxes(input.data.cpu().numpy()[0],0,2)
#cv2.imshow('second_try.png', im)
cv2.imwrite(args.save_folder + 'result_new.png', im)
