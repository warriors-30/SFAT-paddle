import copy
import paddle
import numpy as np
import paddle.nn.functional as F
import attack_generator as attack
from paddle import nn
from paddle.io import DataLoader, Dataset
from tqdm import tqdm

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, alg, anchor, anchor_mu, local_rank, aysn=False, method='AT'):
        self.args = args
        self.trainloader, self.testloader = self.train_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.loss.CrossEntropyLoss(reduction="mean").to(self.device)
        self.alg = alg
        self.anchor = anchor
        self.anchor_mu = anchor_mu
        self.local_rank = local_rank
        self.asyn = aysn
        self.method = method

    def train_test(self, dataset, idxs):
        """
        Returns train and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(1.0*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=self.args.lr,
                                         weight_decay=1e-4)

        for iter in tqdm(range(self.args.local_ep)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader())):
                #images, labels = images.to(self.device), labels.to(self.device)

                optimizer.clear_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_at(self, model, global_round):
        
        epoch_loss = []
        index = 0.0
        index_pgd =0
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=self.args.lr,
                                                  momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=self.args.lr,
                                              weight_decay=1e-4)
        timestep = 0
        max_local_train = self.args.local_ep
        tau = 0
        
        if self.args.dataset == 'cifar-10':
            eps = 8/255
            sts = 2/255
        if self.args.dataset == 'svhn':
            eps = 4/255
            sts = 1/255
        if self.args.dataset == 'cifar-100':
            eps = 8/255
            sts = 2/255   

        total, correct = 0.0, 0.0
        lop = self.args.local_ep
        
        for iter in tqdm(range(lop)):
            batch_loss = []
            index = 0.0
            index_pgd = 0
            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader())):
                #images, labels = images.to(self.device), labels.to(self.device)
                if self.method == 'AT':
                    x_adv, ka = attack.PGD(model,images,labels,eps,sts,self.args.num_steps,loss_fn="cent",category="Madry",rand_init=True)
                elif self.method == 'ST':
                    x_adv, ka = images, 0
                  
                model.train()
                optimizer.clear_grad()
                log_probs = model(x_adv)
                
                pred_labels = paddle.argmax(log_probs, 1)
                #pred_labels = pred_labels.view(-1)
                correct += paddle.sum(paddle.equal(pred_labels, labels)).item()
                total += len(labels)
                
                if self.method == 'AT':
                    loss = self.criterion(log_probs, labels)
                elif self.method == 'ST':
                    loss = self.criterion(log_probs, labels)
                    ka -= loss.sum().item()
                    
                index_pgd += sum(ka)
                ka = -(loss.sum().item())*len(x_adv)
                
                if self.alg == 'FedProx' and global_round > 0:
                    proximal_term = 0
                    for w, w_t in zip(model.parameters(), self.anchor.parameters()) :
                        # update the proximal term 
                        #proximal_term += torch.sum(torch.abs((w-w_t)**2))
                        proximal_term += (w-w_t).norm(2)
                    loss = loss + (self.anchor_mu/2)*proximal_term
                
                loss.backward()
                optimizer.step()

                if self.method == 'ST':
                    index += ka
                else:
                    index = index + ka
  
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
                
                    
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if timestep == 0 and len(self.local_rank) >= 1 and self.asyn == True and global_round>0:
                selfrank = index / len(self.trainloader)
                ipx_sorted = np.sort(self.local_rank)
                ipx_idx = int(self.args.num_users * self.args.frac)
                ipx_value = ipx_sorted[int(ipx_idx*(1-min(1,global_round/(0.15*self.args.epochs))))]
                if selfrank < ipx_value and max_local_train == self.args.local_ep:
                    print(1)
                    max_local_train = int(self.args.es*max_local_train)
            timestep = timestep+1
            if timestep > max_local_train and len(self.local_rank) >= 1 and self.asyn == True and max_local_train != self.args.local_ep:
                break

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), index / len(self.trainloader), correct/total, index_pgd/ len(self.trainloader)

    def update_weights_scaffold(self, model, c_global_model, c_local_model, global_round):
        model.train()

        epoch_loss = []
        index = 0.0
        index_pgd = 0
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=self.args.lr,
                                                  momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=self.args.lr,
                                              weight_decay=1e-4)

        c_global_para = copy.deepcopy(c_global_model.state_dict())
        global_model_para = copy.deepcopy(model.state_dict())
        cnt = 0
        c_local_para = copy.deepcopy(c_local_model.state_dict())
        local_lr = 1e-4
        timestep = 0
        max_local_train = self.args.local_ep
        tau = 0

        if self.args.dataset == 'cifar-10':
            eps = 8 / 255
            sts = 2 / 255
        if self.args.dataset == 'svhn':
            eps = 4 / 255
            sts = 1 / 255
        if self.args.dataset == 'cifar-100':
            eps = 8 / 255
            sts = 2 / 255

        total, correct = 0.0, 0.0
        for iter in tqdm(range(self.args.local_ep)):
            batch_loss = []
            index = 0.0
            index_pgd = 0
            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader())):
                #images, labels = images.to(self.device), labels.to(self.device)
                if self.method == 'AT':
                    x_adv, ka = attack.PGD(model, images, labels, eps, sts, self.args.num_steps, loss_fn="cent",
                                           category="Madry", rand_init=True)
                elif self.method == 'ST':
                    x_adv, ka = images, 0

                model.train()
                optimizer.clear_grad()
                log_probs = model(x_adv)

                pred_labels = paddle.argmax(log_probs, 1)
                # pred_labels = pred_labels.view(-1)
                correct += paddle.sum(paddle.equal(pred_labels, labels)).item()
                total += len(labels)

                if self.method == 'AT':
                    loss = self.criterion(log_probs, labels)
                elif self.method == 'ST':
                    loss = self.criterion(log_probs, labels)
                    ka -= loss.sum().item()

                index_pgd += sum(ka)
                ka = -(loss.sum().item()) * len(x_adv)

                if self.alg == 'FedProx' and global_round > 0:
                    proximal_term = 0
                    for w, w_t in zip(model.parameters(), self.anchor.parameters()):
                        # update the proximal term
                        # proximal_term += torch.sum(torch.abs((w-w_t)**2))
                        proximal_term += (w - w_t).norm(2)

                    loss = loss + (self.anchor_mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

                if self.method == 'ST':
                    index += ka
                else:
                    index = index + ka

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())

                net_para = copy.deepcopy(model.state_dict())
                for key in net_para:
                    net_para[key] = net_para[key] - local_lr * (c_global_para[key] - c_local_para[key])
                model.set_state_dict(net_para)
                cnt += 1

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if timestep == 0 and len(self.local_rank) >= 1 and self.asyn == True and global_round > 0:
                selfrank = index / len(self.trainloader)
                ipx_sorted = np.sort(self.local_rank)
                ipx_idx = int(self.args.num_users * self.args.frac)
                ipx_value = ipx_sorted[int(ipx_idx * (1 - min(1, global_round / (0.15 * self.args.epochs))))]
                if selfrank < ipx_value and max_local_train == self.args.local_ep:
                    print(1)
                    max_local_train = int(self.args.es * max_local_train)
            timestep = timestep + 1
            if timestep > max_local_train and len(
                    self.local_rank) >= 1 and self.asyn == True and max_local_train != self.args.local_ep:
                break

        c_new_para = copy.deepcopy(c_local_model.state_dict())
        c_delta_para = copy.deepcopy(c_local_model.state_dict())
        net_para = copy.deepcopy(model.state_dict())
        for key in net_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (
                    global_model_para[key] - net_para[key]) / (cnt * local_lr)
            c_delta_para[key] = c_new_para[key] - c_local_para[key]
        c_local_model.set_state_dict(c_new_para)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), index / len(
            self.trainloader), correct / total, c_local_model, c_delta_para, index_pgd / len(self.trainloader)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader()):
            #images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            pred_labels = paddle.argmax(outputs, 1)
            #pred_labels = pred_labels.view(-1)
            correct += paddle.sum(paddle.equal(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = paddle.device.set_device('gpu:0') if args.gpu else paddle.device.set_device('cpu')
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(tqdm(testloader())):
        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        pred_labels = paddle.argmax(outputs, 1)
        #pred_labels = pred_labels.view(-1)
        correct += paddle.sum(paddle.equal(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
