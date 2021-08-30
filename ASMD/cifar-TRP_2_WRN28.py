'''
Training script for CIFAR-10/100
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from collections import OrderedDict
import numpy as np
#import matplotlib.pyplot as plt

#from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils import Bar, AverageMeter, accuracy, mkdir_p
from decompose_2_vgg import VH_decompose_model,channel_decompose, network_decouple, \
    EnergyThreshold, ValueThreshold, LinearRate


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 100],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint_res20', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='./checkpoint_wrn28_10/checkpoint.pth.tar', type=str, metavar='PATH',   # ./checkpoint/checkpoint.pth.tar
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='wrn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=1, action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#decouple options
parser.add_argument('--decouple-period', '-dp', type=int, default=1, help='set the period of TRP')
parser.add_argument('--trp', dest='trp', default=0, help='set this option to enable TRP during training', action='store_true')
parser.add_argument('--type', type=str, help='the type of decouple', choices=['NC','VH','ND'], default='VH')
parser.add_argument('--nuclear-weight', type=float, default=None, help='The weight for nuclear norm regularization')
parser.add_argument('--retrain', dest='retrain', default=1, help='wether retrain from a decoupled model, only valid when evaluation is on', action='store_true')
parser.add_argument('--finetune_epoch', default=120, type=int, metavar='N',help='number of total epochs to run')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device_id = int(args.gpu_id)
use_cuda = torch.cuda.is_available() and device_id >= 0

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
period = args.decouple_period
assert period >= 1

DEBUG = False # debug option for singular value

# set decouple method
if args.type == 'VH':
    f_decouple = VH_decompose_model
elif args.type == 'NC':
    f_decouple = channel_decompose
elif args.type == 'ND':
    f_decouple = network_decouple
else:
    raise NotImplementedError('no such decouple type %s' % args.type)

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    
    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    # model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda(torch.device('cuda:1'))
    model = model.cuda()
    cudnn.benchmark = True
    print('   Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    #torch.save(model,'tempolary.pth')
    #new_model = torch.load('tempolary.pth')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
#        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
#    else:
#        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
#        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    
    look_up_table = get_look_up_table(model)
    print(look_up_table)
    print(model)

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

        print(' Start decomposition:')
         
        # set different threshold for model compression and test accuracy
        thresholds = [1e-8] if args.type != 'ND' else [0.85]   #
        sigma_criterion = ValueThreshold if args.type != 'ND' else EnergyThreshold
        T = np.array(thresholds)
        ff_new = np.zeros(T.shape)
        acc = np.zeros(T.shape)
        pp_new = np.zeros(T.shape)
        pp = np.zeros(T.shape)
        ff = np.zeros(T.shape)
        fr = np.zeros(T.shape)
        pr = np.zeros(T.shape)
        #torch.save(model, 'net.pth')
        #result = 'result.pth' if not args.retrain else 'result-retrain.pth'
        prune_epoch = 1
        finetune = 3    
        for i, t in enumerate(thresholds):
            #test_model = torch.load('net.pth') 
            test_model = model        
            
            ff[i] = show_FLOPs(test_model, look_up_table, input_size=[32, 32], criterion=sigma_criterion(t), type=args.type)
            pp[i] = sum(p.numel() for p in test_model.parameters())/1000000.0
            #######################################################################
            for loop in range(2) :  
                test_model, store_r = f_decouple(test_model, look_up_table, criterion=sigma_criterion(t), train=False, sign = False, prune_epoch= prune_epoch)
                prune_epoch += 1
                print(' Done! test decoupled model')
                test_loss, test_acc = test(testloader, test_model, criterion, start_epoch, use_cuda)
                print(' Test Loss :  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
                acc[i] = test_acc
            
                if args.retrain:
                   finetune_epoch = finetune
                   acc[i] = model_retrain(finetune_epoch, test_model, trainloader, \
                        testloader, criterion, look_up_table, use_cuda)
                finetune += 5  
                #ff_new[i] = show_FLOPs_new(test_model, look_up_table, store_r, input_size=[32, 32], criterion=sigma_criterion(t), type=args.type)
                #fr[i] = ff[i] / ff_new[i]
                #print('flop ratio:')
                #print(fr)
            #######################################################################

            test_model, store_r = f_decouple(test_model, look_up_table, criterion=sigma_criterion(t), train=False, sign = True)
            print(' Done! test decoupled model')
            test_loss, test_acc = test(testloader, test_model, criterion, start_epoch, use_cuda)
            print(' Test Loss :  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
            acc[i] = test_acc
            print('   Total params in decoupled model: %.2fM' % (sum(p.numel() for p in test_model.parameters())/1000000.0))

            if args.retrain:
                finetune_epoch = args.finetune_epoch
                acc[i] = model_retrain(finetune_epoch, test_model, trainloader, \
                     testloader, criterion, look_up_table, use_cuda)

            print('store_r:')
            print(store_r)
            ff_new[i] = show_FLOPs_new(test_model, look_up_table, store_r, input_size=[32, 32], criterion=sigma_criterion(t), type=args.type)
            pp_new[i] = sum(p.numel() for p in test_model.parameters())/1000000.0
            fr[i] = ff[i] / ff_new[i]
            pr[i] = pp[i] / pp_new[i]

            print('flop ratio:')
            print(fr)
            print('prune ratio:')
            print(pr)
            ###############################################################
        torch.save(test_model, 'model.pth.tar')
    #    torch.save(OrderedDict([('acc',acc),('cr', cr)]), result)
        print('accuracy:')
        print(acc)

        return

    # Train and val

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, look_up_table, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
#        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

#    logger.close()
#    logger.plot()
#    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def model_retrain(finetune_epoch, test_model, trainloader, testloader, criterion, look_up_table, use_cuda):
    print(' Retrain decoupled model')
    finetune_epoch = finetune_epoch
    
    best_acc = 0.0
    optimizer = optim.SGD(test_model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    global state
    init_lr = args.lr
    state['lr'] = init_lr
 
    for epoch in range(finetune_epoch):

        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, finetune_epoch, state['lr']))
        train_loss, train_acc = train(trainloader, test_model, criterion, optimizer, look_up_table, epoch, use_cuda)
        test_loss, test_acc = test(testloader, test_model, criterion, epoch, use_cuda)
        best_acc = max(test_acc, best_acc)

    return best_acc

def get_look_up_table(model):
    count = 0
    look_up_table = []
    First_conv = True
    First_conv = False

    for name, m in model.named_modules():
        #print(name)
        #print('*****')
        #print(m)
        #TODO: change the if condition here to select different kernel to decouple
        if isinstance(m, nn.Conv2d) and m.kernel_size != (1,1) and count > 0:  #### change !!!!!!!
        #if isinstance(m, nn.Conv2d) and count > 0:    
            #print(count)
            if First_conv:
                First_conv = False
            else:
                #print(count)
                look_up_table.append(name)

        count += 1
    print(look_up_table)
    return look_up_table

def show_FLOPs(model, look_up_table=[], input_size=None, criterion=None, type='NC'):

    redundancy = OrderedDict()
    comp_rate =[]
    origin_FLOPs = 0.

    #size_wh = [32, 16, 8, 8, 8, 8]  # alexnet
    #size_wh = [32, 32, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2] # vgg16 20M

    i = 0
    for name, m in model.named_modules():
        
        if isinstance(m, nn.Conv2d):   
           p = m.weight.data        
           dim = p.size()
           FLOPs = dim[0]*dim[1]*dim[2]*dim[3]
           redundancy[name] = ('%.3f' % (FLOPs) )
           comp_rate.append(FLOPs)
    
           if 'downsample' in name:    #   make changes
              # a special case for resnet
              output_h = input_size[0]/m.stride[0]
              output_w = input_size[1]/m.stride[1]
           else:
              output_h = input_size[0]
              output_w = input_size[1]  # change !!!!!~~
              #output_h = size_wh[i]
              #output_w = size_wh[i]
           i += 1    
        
           origin_FLOPs += FLOPs*output_h*output_w
           input_size = [output_h, output_w]

        elif isinstance(m, nn.Linear):
           p = m.weight.data        
           dim = p.size()
           FLOPs = dim[0]*dim[1]
           redundancy[name] = ('%.3f' % (FLOPs) )
           comp_rate.append(FLOPs)
           origin_FLOPs += FLOPs
        else :
             continue 

    return origin_FLOPs 
    
def show_FLOPs_new(model, look_up_table=[], store_r=[], input_size=None, criterion=None, type='VH' ):
   
    redundancy = OrderedDict()
    comp_rate =[]    
    origin_FLOPs = 0.
    #decouple_FLOPs = 0.    

    #size_wh = [32, 32, 16, 16, 8,8,8,8,8,8,8,8] # alexnet
    #size_wh = [32, 32, 32, 32, 16, 16, 16, 16, 8,8,8,8, 8,8,8,8, 4,4,4,4, 4,4,4,4, 2,2,2,2, 2,2,2,2]  # vgg16 20M
    i = 0
    for name, m in model.named_modules():
      
        if isinstance(m, nn.Conv2d):
         
           p = m.weight.data
           dim = p.size()
           FLOPs = dim[0]*dim[1]*dim[2]*dim[3]
           #count = 0
           redundancy[name] = ('%.3f' % (FLOPs) )
           comp_rate.append(FLOPs)

           #if name in look_up_table and m.stride == (1,1):
           
           #    VH = p.permute(1,2,0,3).contiguous().view(dim[1]*dim[2],-1)
           #    V, sigma, H =torch.svd(VH, some=True)
           #    item_num = store_r[count]
           #    print(item_num)
           #    print(count)
           #    count = count + 1
           #    new_FLOPs = dim[1]*item_num*dim[2]+dim[0]*item_num*dim[3]    
 
           if 'downsample' in name:    #   make changes
               # a special case for resnet
               output_h = input_size[0]/m.stride[0]
               output_w = input_size[1]/m.stride[1]
           else:
               output_h = input_size[0]
               output_w = input_size[1]
               #output_h = size_wh[i]
               #output_w = size_wh[i]
           i += 1   
           origin_FLOPs += FLOPs*output_h*output_w
           #decouple_FLOPs += new_FLOPs*output_h*output_w
           input_size = [output_h, output_w]
        elif isinstance(m, nn.Linear):
           p = m.weight.data        
           dim = p.size()
           FLOPs = dim[0]*dim[1]
           redundancy[name] = ('%.3f' % (FLOPs) )
           comp_rate.append(FLOPs)
           origin_FLOPs += FLOPs
        else :
             continue 

    return origin_FLOPs

def low_rank_approx(model, look_up_table, criterion, use_trp, type='NC'):
    dict2 = model.state_dict()
    sub=dict()
    #can set m here
    for name in dict2:
        param = dict2[name]
        dim = param.size()
        #print(name)
        #print(param)
        #print(dim)
        model_name = name[:-7] if len(dim) == 4 else ''   # conv:  .weight  
        #print(model_name)
        if len(dim) == 4 and model_name in look_up_table:
            if type=='VH':
                VH = param.permute(1, 2, 0, 3).contiguous().view(dim[1]*dim[2], -1)
                try:
                    V, sigma, H = torch.svd(VH, some=True)
                    # print(sigma.size())
                    H = H.t()
                    # remain large singular value
                    valid_idx = criterion(sigma)
                    V = V[:, :valid_idx].contiguous()
                    sigma = sigma[:valid_idx]
                    dia = torch.diag(sigma)
                    H = H[:valid_idx, :]
                    if use_trp:
                        new_VH = (V.mm(dia)).mm(H)
                        new_VH = new_VH.contiguous().view(dim[1], dim[2], dim[0], dim[3]).permute(2, 0, 1, 3)
                        dict2[name].copy_(new_VH)
                    subgradient = torch.mm(V, H)
                    subgradient = subgradient.contiguous().view(dim[1], dim[2], dim[0], dim[3]).permute(2, 0, 1, 3)
                    sub[model_name] = subgradient
                    #dict2[name].copy_(param)   # add
                except:
                    sub[model_name] = 0.0
                    dict2[name].copy_(param)
            elif type == 'NC':
                NC = param.contiguous().view(dim[0], -1)
                try:
                    N, sigma, C = torch.svd(NC, some=True)
                    # print(sigma.size())
                    C = C.t()
                    # remain large singular value
                    valid_idx = criterion(sigma)
                    N = N[:, :valid_idx].contiguous()
                    sigma = sigma[:valid_idx]
                    dia = torch.diag(sigma)
                    C = C[:valid_idx, :]
                    if use_trp:
                        new_NC = (N.mm(dia)).mm(C)
                        new_NC = new_NC.contiguous().view(dim[0], dim[1], dim[2], dim[3])
                        dict2[name].copy_(new_NC)
                    subgradient = torch.mm(N, C)
                    subgradient = subgradient.contiguous().view(dim[0], dim[1], dim[2], dim[3])
                    sub[model_name] = subgradient
                except:
                    sub[model_name] = 0.0
                    dict2[name].copy_(param)
            else:
                # network decouple approximation
                tmp = param.clone()
                tmp_sub = param.clone()
                valid_idx = 0

                for i in range(dim[0]):
                    W = param[i, :, :, :].view(dim[1], -1)
                    try:
                        U, sigma, V = torch.svd(W, some=True)
                        V = V.t()
                        valid_idx = criterion(sigma)
                        U = U[:, :valid_idx].contiguous()
                        V = V[:valid_idx, :].contiguous()
                        sigma = sigma[:valid_idx]
                        dia = torch.diag(sigma)
                        if use_trp:
                            new_W = (U.mm(dia)).mm(V)
                            new_W = new_W.contiguous().view(dim[1], dim[2], dim[3])
                            tmp[i, :, :, :] = new_W[...]
                        subgradient = torch.mm(U, V)
                        subgradient = subgradient.contiguous().view(dim[1], dim[2], dim[3])
                        tmp_sub[i, :, :, :] = subgradient[...]
                    except Exception as e:
                        print(e)
                        tmp_sub[i, :, :, :] = 0.0
                        tmp[i, :, :, :] = param[i, :, :, :]

                dict2[name].copy_(tmp)
                sub[model_name] = tmp_sub
            #print(param)    
        else:
            dict2[name].copy_(param)
            #print(param)
    model.load_state_dict(dict2)
    #print(param)
    #assert 1 == 0
    return model, sub


def train(trainloader, model, criterion, optimizer, look_up_table, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        #if batch_idx % period == 0:
        #    model, sub = low_rank_approx(model, look_up_table, criterion=EnergyThreshold(0.9), use_trp=args.trp, type=args.type)      

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
         
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        # apply nuclear norm regularization
        if args.nuclear_weight is not None and batch_idx % period == 0:
            for name, m in model.named_modules():
                if name in look_up_table:
                    m.weight.grad.data.add_(args.nuclear_weight*sub[name])

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    DEBUG = False
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
