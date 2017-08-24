import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from layers.modules import MultiBoxLoss

from ssd import build_ssd
from torchvision import transforms

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
parser.add_argument('--save_folder', default='new/', type=str,
                    help='File path to save results')
parser.add_argument('--lr', '--learning-rate', default=1, type=float, help='initial learning rate')
parser.add_argument('--lam', default=1e4, type=float, help='weighting of l2 loss')
parser.add_argument('--refine', default='', type=str, help='when set, the given image is refined')
parser.add_argument('--iterations', default=100, type=int, help='How long the class model shall be optimized')
parser.add_argument('--classes', default=['horse'], nargs='+', help='The class, that shal be recognised in the image')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
lr = args.lr

torch.set_default_tensor_type('torch.cuda.FloatTensor')
img_size = 300

#criterion = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False, True)

net = build_ssd('train', img_size, 21)    # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
net.eval()

criterion = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False, True)




prep = transforms.Compose([
    transforms.Scale((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
postpa = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0],
        std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
        std=[1,1,1]),
    ])
postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    tensor = tensor.cpu()
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(tensor)
    return img




for category in args.classes:
    i = 8729
    category_index = VOC_CLASSES.index(category)
    print('New category: ' + category + ' (' + str(category_index) + ')')
    if args.refine == '':
        input = Variable(torch.zeros(1, 3, img_size, img_size), requires_grad=True)
    else:
        im = np.swapaxes(cv2.imread(args.refine),0,2).astype('f')
        input =Variable((torch.from_numpy(np.expand_dims(im, axis=0)).cuda()), requires_grad=True)

    optimizer = optim.Adam([input], lr=lr)

    for iteration in range(1, args.iterations):
        out = net(input)

        targets = torch.zeros(1,1,4)
        targets[0,0,2] = 1
        targets[0,0,3] = 1
        #targets[0,0,4] = category_index
        #L4:
        targets[0,0,0] = 0.25
        targets[0,0,1] = 0.25
        targets[0,0,2] = 0.75
        targets[0,0,3] = 0.75

        targets = Variable(targets, requires_grad=False)

        #loss_l, loss_c = criterion(out, targets)
        #class_loss = -(out[1][0, :, category_index].mean() + 0.001*out[1][0, :, category_index].max())

        '''for i, l in enumerate(out[2]):
            out[2][i] = (l-targets)**2
        v, i = out[2].sum(1).min(0)
        print(v)
        print(out[2][8693])
        break'''

        l2_loss = args.lam * (input**2).mean()

        '''#L1:
        class_loss = -(out[1][0, 8693, category_index+1] + (out[0][0, 8693, :] - targets).sum())
        loss = l2_loss + class_loss

        #L2:
        class_loss = -(out[1][0, :, category_index+1].mean())
        loss = l2_loss + class_loss

        #L3:
        loss_l, loss_c = criterion(out, targets)
        loss = l2_loss + loss_l + loss_c'''

        #L4: L3 with see target
        #L5: L1 with see target

        #L6: L1 without location:
        class_loss = -(out[1][0, i, category_index+1])
        loss = l2_loss + class_loss

        optimizer.zero_grad()
        print(str(iteration) + ': loss: ' + str(loss.data[0]))

        '''if (iteration % round(args.iterations/3)) == 0:
            lr = lr / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Adjusted learning rate to ' + str(lr))
            print('iteration: ' + str(iteration) + ' loss: ' + str(loss))
	'''

        loss.backward()
        optimizer.step()

    im = postp(input.data.clone().squeeze())
    im.save(args.save_folder + str(category) + '_l6_box' + str(i) + '_' + str(args.iterations) + '_' + str(lr) + '_lam' + str(args.lam) + '.png')
    #cv2.imwrite(args.save_folder + 'result_' + str(category) + '.png', im)
