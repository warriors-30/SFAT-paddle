import os
import copy
import time
import paddle
import pickle
import numpy as np
import paddle.nn.functional as F
import matplotlib.pyplot as plt
import attack_generator as attack
from models import *
from paddle import nn
from tqdm import tqdm
from logger import Logger
from utils import get_dataset
from options import args_parser
from numpy.random import shuffle
from update import test_inference
from matplotlib.pyplot import title
from paddle.io import DataLoader

# Save checkpoint
def save_checkpoint(state, checkpoint='../centralized_AT_result', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    paddle.save(state, filepath)


if __name__ == '__main__':
    args = args_parser()
    #device = 'gpu:0' if args.gpu else 'cpu'
    device = paddle.device.set_device('gpu:0') if args.gpu else paddle.device.set_device('cpu')

    seed = args.seed
    paddle.seed(seed)
    np.random.seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True

    # Store path
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # build model
    if args.modeltype == 'NIN':
        global_model = NIN()
    elif args.modeltype == 'SmallCNN':
        global_model = SmallCNN()
    elif args.modeltype == 'resnet18':
        global_model = ResNet18()

    print('==> CENTER')
    title = 'CENTER'
    logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Global Epoch', 'Local Epoch', 'Epoch', 'Natural Test Acc', 'PGD20 Acc'])


    if args.dataset == 'cifar-10':
        eps = 8/255
        sts = 2/255
    if args.dataset == 'svhn':
        eps = 4/255
        sts = 1/255
    if args.dataset == 'cifar-100':
        eps = 8/255
        sts = 2/255

    # Set the model to train and send it to device.
    global_model.to(device)
    print(global_model)

    global_best_natural = 0
    global_best_pgd = 0
    best_epoch = 0

    # Training

    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = paddle.optimizer.Momentum(parameters=global_model.parameters(), learning_rate=args.lr,
                                              momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = paddle.optimizer.Adam(parameters=global_model.parameters(), learning_rate=args.lr,
                                          weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)

    criterion = paddle.nn.CrossEntropyLoss(reduction="mean").to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(tqdm(trainloader())):
            #images, labels = images.to(device), labels.to(device)
            x_adv, _ = attack.PGD(global_model,images,labels,eps,sts,args.num_steps,loss_fn="cent",category="Madry",rand_init=True)
            
            global_model.train()
            optimizer.clear_grad()
            outputs = global_model(x_adv)
            if args.train_method == 'AT':
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

        _, test_nat_acc = attack.eval_clean(global_model, testloader)
        _, test_pgd20_acc = attack.eval_robust(global_model, testloader, perturb_steps=20, epsilon=eps, step_size=sts,loss_fn="cent", category="Madry", random=True)

        print('Nat Test Acc: {:.2f}%'.format(100*test_nat_acc))
        print('PGD-20 Test Acc: {:.2f}%'.format(100*test_pgd20_acc))

        if test_pgd20_acc >= global_best_pgd:
            global_best_pgd = test_pgd20_acc
            global_best_natural = test_nat_acc
            best_epoch = epoch
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': global_model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                'test_pgd20_acc': test_pgd20_acc,
            },checkpoint=args.out_dir,filename='bestpoint.pth.tar')
        
        logger_test.append([args.epochs, args.local_ep, epoch, test_nat_acc, test_pgd20_acc])

    # final testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
