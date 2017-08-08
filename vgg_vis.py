import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from layers.modules import MultiBoxLoss

import torchvision
from ssd import build_ssd

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Class Model Visualization for the VGG network')
parser.add_argument('--save_folder', default='vis/', type=str,
                    help='File path to save results')
parser.add_argument('--lr', '--learning-rate', default=5, type=float, help='initial learning rate')
parser.add_argument('--refine', default='', type=str, help='when set, the given image is refined')
parser.add_argument('--iterations', default=60000, type=int, help='How long the class model shall be optimized')
parser.add_argument('--classes', default=['goose'], nargs='+', help='The class, that shal be recognised in the image')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
lr = args.lr

torch.set_default_tensor_type('torch.cuda.FloatTensor')


#criterion = torch.nn.CrossEntropyLoss().cuda()
net = torchvision.models.vgg19(pretrained=True).cuda()

'''normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
path = 'vis_upload/cats.jpg'
input = torch.from_numpy(np.swapaxes(cv2.imread(path), 0, 2).astype('f')[:, 0:244, 0:244])/255
input = normalize.__call__(input)
input = Variable(input, requires_grad=True).cuda()
input = input.unsqueeze(0)
print(input.size())
output = net(input)
print(output)'''



for category in args.classes:
    category_index = 99 #goose index
    print('New category: ' + category + ' (' + str(category_index) + ')')
    if args.refine == '':
        input = Variable(torch.zeros(1, 3, 244, 244), requires_grad=True)
    else:
        im = np.swapaxes(cv2.imread(args.refine),0,2).astype('f')
        input =Variable((torch.from_numpy(np.expand_dims(im, axis=0)).cuda()), requires_grad=True)

    #targets = torch.LongTensor([category_index]).cuda()
    #targets = Variable(targets, requires_grad=False)

    optimizer = optim.Adam([input], lr=lr)

    for iteration in range(1, args.iterations):
        out = net(input)

        optimizer.zero_grad()

        #Difference to ssd:
        loss = 1/out[0, category_index]

        if iteration % 2000 == 0:
            print('loss' + str(loss))

        #Adopting learning rate:
        if (iteration % round(args.iterations/4)) == 0:
            lr = lr / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            print('Adjusted learning rate to ' + str(lr))
            print('iteration: ' + str(iteration) + ' loss: ' + str(loss))

        loss.backward()
        optimizer.step()

    im = np.swapaxes(input.data.cpu().numpy()[0],0,2)
    cv2.imwrite(args.save_folder + 'test_result_' + str(category) + '.png', im)
