import paddle
import argparse
import paddle.vision
import paddle.nn as nn
import attack_generator as attack
from paddle.vision import datasets,transforms
from models import *
from utils import custom_cifar10,custom_cifar100,custom_svhn
from paddle.io import DataLoader

parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="resnet18", help="decide which network to use,choose from SmallCNN,resnet18,NIN")
parser.add_argument('--dataset', type=str, default="svhn", help="choose from cifar10,svhn,cifar100")
parser.add_argument('--model_path', default="./bestpoint.pth.tar", help='model for white-box attack evaluation')
parser.add_argument('--method',type=str,default='dat',help='select attack setting following DAT or TRADES')

args = parser.parse_args()
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

device = paddle.device.set_device('gpu:0') if args.gpu else paddle.device.set_device('cpu')

num_c = 10
print('==> Load Test Data')
if args.dataset == "cifar-10":
    data_dir = '../data/cifar-10/'
    test_dataset = custom_cifar10(data_dir, mode='test', transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    num_c = 10
if args.dataset == "svhn":
    data_dir = '../data/svhn/'
    test_dataset = custom_svhn(data_dir, split='test', transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
if args.dataset == "cifar-100":
    data_dir = '../data/cifar-100/'
    test_dataset = custom_cifar100(data_dir, mode='test', transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    num_c = 100

print('==> Load Model')
if args.net == "SmallCNN":
    model = SmallCNN().to(device)
    net = "smallcnn"
if args.net == "resnet18":
    model = ResNet18(num_classes=num_c).to(device)
    net='resnet18'
if args.net == "NIN":
    model = NIN().to(device)
    net = "NIN"

print(net)
print(args.model_path)
model.set_state_dict(paddle.load(args.model_path)['state_dict'])
print(paddle.load(args.model_path)['epoch'])
print('==> Evaluating Performance under White-box Adversarial Attack')

loss, test_nat_acc = attack.eval_clean(model, test_loader)
print('Natural Test Accuracy: {:.2f}%'.format(100. * test_nat_acc))
if args.method == "dat":
    # Evalutions the same as DAT.
    if args.dataset != "svhn":
      loss, fgsm_wori_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=8/255, step_size=8/255,loss_fn="cent", category="Madry",random=True)
      print('FGSM without Random Start Test Accuracy: {:.2f}%'.format(100. * fgsm_wori_acc))
      loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=8/255, step_size=2/255,loss_fn="cent", category="Madry", random=True)
      print('PGD20 Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))
    else:
      loss, fgsm_wori_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=4/255, step_size=1/255,loss_fn="cent", category="Madry",random=True)
      print('FGSM without Random Start Test Accuracy: {:.2f}%'.format(100. * fgsm_wori_acc))
      loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=4/255, step_size=1/255,loss_fn="cent", category="Madry", random=True)
      print('PGD20 Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))