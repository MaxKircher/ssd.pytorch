import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

#some more imports
import os
cwd = os.getcwd()
from os.path import join as pj
from PIL import Image

#torch imports
import torch
from torch import optim
from torch.autograd import Variable
# import torch.nn.functional as F

#torchvision imports
import torchvision.models as models
from torchvision import transforms

#IPython imports
from IPython import display

parser = argparse.ArgumentParser(description='Class Model Visualization for the VGG network')
parser.add_argument('--save_folder', default='vis/', type=str,
                    help='File path to save results')
parser.add_argument('--lr', '--learning-rate', default=5, type=float, help='initial learning rate')
parser.add_argument('--refine', default='', type=str, help='when set, the given image is refined')
parser.add_argument('--iterations', default=500, type=int, help='How long the class model shall be optimized')
parser.add_argument('--classes', default=['goose'], nargs='+', help='The class, that shal be recognised in the image')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
lr = args.lr

torch.set_default_tensor_type('torch.cuda.FloatTensor')


#criterion = torch.nn.CrossEntropyLoss().cuda()
net = torchvision.models.vgg11(pretrained=True).cuda()

'''normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
path = 'vis_upload/cats.jpg'
input = torch.from_numpy(np.swapaxes(cv2.imread(path), 0, 2).astype('f')[:, 0:244, 0:244])/255
input = normalize.__call__(input)
input = Variable(input, requires_grad=True).cuda()
input = input.unsqueeze(0)
print(input.size())
output = net(input)
print(output)'''


# define pre and post processing for images using the torchvision helper functions
img_size = 224
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
    img = postpb(t)
    return img

#define image variable to optimise
opt_img = Variable(1e-3 * torch.zeros(1, 3, img_size, img_size), requires_grad=True)
if args.refine != '':
    im = np.swapaxes(cv2.imread(args.refine),0,2).astype('f')
    opt_img =Variable((torch.from_numpy(np.expand_dims(im, axis=0)).cuda()), requires_grad=True)

#set all the optimisation parameters
max_iter = args.iterations #maximum number of iterations to take
lr = args.lr #learning rate of the optimisation method
lam = 1e3 #weight on the L2 regularisation of the input image
optimizer = optim.SGD([opt_img], lr=lr)
target_class = 99

n_iter = 0
while n_iter <= max_iter:
    optimizer.zero_grad()
    output = net(opt_img)
    class_loss = -output[0,target_class]
    #l2_loss = Variable(torch.zeros(1))
    l2_loss = lam * (opt_img**2).mean()
    loss = class_loss + l2_loss
    #print loss and show intermediate result image
    print('iter: %d, total loss: %.3f, class loss: %.3f, l2 loss: %.3f'%(n_iter, loss.data[0], class_loss.data[0], l2_loss.data[0]))
    #plt.imshow(postp(opt_img.data.clone().squeeze()))
    #display.display(plt.gcf())
    #display.clear_output(wait=True)
    loss.backward()
    optimizer.step()
    n_iter +=1

postp(opt_img.data.clone().squeeze()).save('new/goose_min_lam1e3_' + str(max_iter) + '_' + str(lr) + '.png')
#cv2.imwrite(args.save_folder + 'cup' + '.png', postp(opt_img.data.clone().squeeze()))


#python vgg_vis.py --iterations 100000 --lr 10000 --refine vis/test_result_goose.png
#python vgg_vis.py --iterations 500000 --lr 100 --refine vis/ref-ref-test_result_goose.png





'''for category in args.classes:
    category_index = 99 #goose index
    print('New category: ' + category + ' (' + str(category_index) + ')')
    if args.refine == '':
        input = Variable(torch.zeros(1, 3, 224, 224), requires_grad=True)
    else:
        im = np.swapaxes(cv2.imread(args.refine),0,2).astype('f')
        input =Variable((torch.from_numpy(np.expand_dims(im, axis=0)).cuda()), requires_grad=True)

    #targets = torch.LongTensor([category_index]).cuda()
    #targets = Variable(targets, requires_grad=False)

    #optimizer = optim.Adam([input], lr=lr)

    for iteration in range(1, args.iterations):
        out = net(input)
        #print(out.size())
        #optimizer.zero_grad()

        #Difference to ssd:
        #loss = out[0, category_index]

        #if iteration % 2000 == 0:
        print('loss' + str(out[0, category_index]))
        #print(out.volatile)

        #Adopting learning rate:
        if (iteration % round(args.iterations/3)) == 0:
            lr = lr / 10
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = lr

            print('Adjusted learning rate to ' + str(lr))
            #print('iteration: ' + str(iteration) + ' loss: ' + str(loss))

        #loss.backward()
        i = torch.zeros(1, 1000)
        i[0,category_index] = 1
        out.backward(i)

        #grad = input.grad
        #d=grad.data
        #print(grad)
        #loss.backward()
        input = Variable((input+lr*input.grad).data, requires_grad=True, volatile=False)

        #input.grad.data.zero_()

        #grad = torch.autograd.grad(input, out)



        #optimizer.step()



    im = np.swapaxes(input.data.cpu().numpy()[0],0,2)
    cv2.imwrite(args.save_folder + 'small' + str(category) + '.png', im)'''
