import paddle
import numpy as np
from models import *
from tqdm import tqdm
#from torch.autograd import Variable

def PGD(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    Kappa = paddle.ones((len(data),))
    if category == "trades":
        x_adv = data.detach() + 0.001 * paddle.randn(data.shape).detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        x_adv = data.detach() + paddle.uniform(data.shape, min=-epsilon, max=epsilon) if rand_init else data.detach()
        x_adv = paddle.clip(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.stop_gradient=False
        output = model(x_adv)
        predict = output.argmax(1, keepdim=True)
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        for param in model.parameters():
            param.clear_grad()
        if loss_fn == "cent":
            loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
        if loss_fn == "kl":
            #criterion_kl = nn.KLDivLoss(size_average=False)
            criterion_kl = nn.KLDivLoss(reduction='sum')
            loss_adv = criterion_kl(F.log_softmax(output, axis=1),F.softmax(nat_output, axis=1))
        loss_adv.backward()
        # Update adversarial data
        x_adv = x_adv.detach() + step_size * x_adv.grad.sign()
        diff = paddle.clip(x_adv - data, min=-epsilon, max=epsilon)  # gradient projection
        x_adv = paddle.clip(data + diff, min=0.0, max=1.0).detach()  # to stay in image range [0,1]
    return x_adv, Kappa

def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with paddle.no_grad():
        for data, target in tqdm(test_loader()):
            #data, target = data.cuda(), target.cuda()
            output = model(data)
            #test_loss += F.cross_entropy(output, target, size_average=False).item()
            test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.equal(target.reshape(pred.shape)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    test_loss = 0
    correct = 0
    #with torch.enable_grad():
    for data, target in tqdm(test_loader()):
        #data, target = data.cuda(), target.cuda()
        x_adv, _ = PGD(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=random)
        output = model(x_adv)
        #test_loss += F.cross_entropy(output, target, size_average=False).item()
        test_loss += F.cross_entropy(output, target).item()
        pred = output.argmax(1, keepdim=True)
        correct += pred.equal(target.reshape(pred.shape)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

