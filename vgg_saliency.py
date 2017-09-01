import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

vgg_classes = {'goose': (99, 100), 'dogs': (152, 268), 'bullet_train': (466, 467), 'electric_locomotive': (547, 548), 'dining_table': (532, 533)}


parser = argparse.ArgumentParser(description='Class Model Visualization for the VGG network')
parser.add_argument('input', help='The image, for which the saliency map shall be computed')
parser.add_argument('--save_folder', default='sal_comp/', type=str,
                    help='File path to save results')
parser.add_argument('--lr', '--learning-rate', default=5, type=float, help='initial learning rate')
parser.add_argument('--refine', default='', type=str, help='when set, the given image is refined')
parser.add_argument('--iterations', default=60000, type=int, help='How long the class model shall be optimized')
parser.add_argument('--classes', default=['dogs'], nargs='+', help='The class, that shal be recognised in the image')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
lr = args.lr

torch.set_default_tensor_type('torch.cuda.FloatTensor')


#criterion = torch.nn.CrossEntropyLoss().cuda()
net = torchvision.models.vgg11(pretrained=True).cuda()

#152-268: dogs
# 466: 'bullet train, bullet',
# 547: 'electric locomotive',
#  532: 'dining table, board',


for category in args.classes:
    category_indices = vgg_classes[category]
    print('New category: ' + category + ' (' + str(category_indices) + ')')

    #Get a 300x300 image out of the given image
    im = np.swapaxes(cv2.imread(args.save_folder+args.input),0,2).astype('f')
    x_off = int((np.size(im, 1)-224)/2)
    y_off = int((np.size(im, 2)-224)/2)
    im = im[:,x_off:224+x_off,y_off:224+y_off]
    input = Variable((torch.from_numpy(np.expand_dims(im, axis=0)).cuda()), requires_grad=True)



    out = net(input)

    #print('loss' + str(out[0, category_indices]))

    loss = -(out[0, category_indices[0]:category_indices[1]].max())

    loss.backward()

    '''i = torch.zeros(1, 1000)
    i[0,category_index] = 1
    out.backward(i)'''

    #input = Variable((input+lr*input.grad).data, requires_grad=True, volatile=False)
    g = input.grad
    #print(g)
    map = g.data.cpu().numpy()[0]
    map = map.max(0)
    # Normalize, so gradients are visible:
    map = 255*map/map.max()
    vis = np.swapaxes(input.data.cpu().numpy()[0]+map, 0, 2)
    cv2.imwrite(args.save_folder + args.input.split('.')[0] + '_saliency_' + str(category) + '.png', np.fliplr(np.rot90(map, k=3)))
    cv2.imwrite(args.save_folder + args.input.split('.')[0] + '_both_' + str(category) + '.png', vis)
    #cv2.imwrite(args.save_folder + 'saliency' + str(category) + '.png', map)








    #im = np.swapaxes(input.data.cpu().numpy()[0],0,2)
    #cv2.imwrite(args.save_folder + 'ref-ref-ref-test_result_' + str(category) + '.png', im)
