# -*- coding: utf-8 -*-

import os
import sys
import copy
from tqdm import tqdm

import torch
import torchvision
from torchsummary import summary
import torchvision.transforms as transforms

# from vgg import VGG16
import config


def return_accuracy(outputs, targets):
    print("len(outputs):", len(outputs), "len(targets):", len(targets))
    print("outputs[0].shape:", outputs[0].shape)
    print("argmax(outputs[0]).shape:", torch.argmax(outputs[0], dim=1).shape)
    print("targets[0].shape", targets[0].shape)
    
    accuracy = []
    for output, target in zip(outputs, targets):
        print("output.shape:", output.shape, "target.shape:", target.shape)
        acc = torch.sum(torch.argmax(output, dim=1) == target) / len(target)
        accuracy.append(acc)
    return accuracy

class MultiTaskLoss(torch.nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        
        self.bceloss = torch.nn.CrossEntropyLoss().to(config.device)
        
    def regularize(self, curr_model, prev_model, alpha=config.alpha, beta=config.beta):

        if prev_model == None:
            return 0
        
        fc_loss = 0.0
        conv_loss = 0.0

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for name, param in curr_model.named_parameters():

            if name.startswith("fc"):
                fc_loss += torch.norm(param)
            else:
                old_param_name = name[18:]

                for o_name, o_param in prev_model.named_parameters():
                    if o_name == old_param_name:
                        if config.regularization == 'l2_norm':
                            loss = torch.norm(param - o_param)
                        elif config.regularization == 'cosine':
                            loss = cos(param, o_param)
                        else:
                            print("Unkown regularization..\ncheck config")

                        conv_loss += loss

        return  0.5 * alpha * conv_loss + 0.5 * beta * fc_loss

    def forward(self, preds, targets, curr_model, prev_model):
        loss = []
        for i in range(config.num_tasks):
            task_loss = self.bceloss(preds[i].view(-1, config.num_classes), targets[:,i])
            loss.append(task_loss)
        
        total_loss = torch.Tensor([0])
        total_loss = total_loss.to(config.device)
        for i in range(config.num_tasks):
            total_loss += loss[i]

        reg_loss = self.regularize(curr_model, prev_model)
        total_loss += reg_loss

        return loss, reg_loss, total_loss




class OWM_ConvNet_3(torch.nn.Module):

    def __init__(self, inputsize):
        super(OWM_ConvNet_3, self).__init__()
        ncha, size, _ = inputsize
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False)
        #self.c4 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=False)
        
        return

    def forward(self, x,):
        h_list = []
        x_list = []

        # Gated
        x = self.padding(x)
        x_list.append(torch.mean(x, 0, True))
        con1 = self.drop1(self.relu(self.c1(x)))
        con1_p = self.maxpool(con1)

        con1_p = self.padding(con1_p)
        x_list.append(torch.mean(con1_p, 0, True))
        con2 = self.drop1(self.relu(self.c2(con1_p)))
        con2_p = self.maxpool(con2)

        con2_p = self.padding(con2_p)
        x_list.append(torch.mean(con2_p, 0, True))
        con3 = self.drop1(self.relu(self.c3(con2_p)))
        con3_p = self.maxpool(con3)

        return con3_p


class Network(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(Network, self).__init__()
        
        if config.model == 'owm_3':
            self.flatten_shape = 256*4*4
            self.feature_extractor = OWM_ConvNet_3((config.num_channels, config.image_shape, config.image_shape)).to(config.device)
            print(self.feature_extractor)
        elif config.model == 'owm_5':
            self.flatten_shape = 1024*6*6
            self.feature_extractor = OWM_ConvNet_5((config.num_channels, config.image_shape, config.image_shape)).to(config.device)
        elif config.model == 'owm_8':
            self.flatten_shape = 1024*4*4
            self.feature_extractor = OWM_ConvNet_8((config.num_channels, config.image_shape, config.image_shape)).to(config.device)
        else:
            print("Unknown model")
        
        self.softmax = torch.nn.Softmax(dim=1)
       
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.flatten_shape, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=config.num_classes),  # Because all attributes are binary
        )
        
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.flatten_shape, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=config.num_classes), 
        )
        
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.flatten_shape, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=config.num_classes),  
        )
        
        self.fc4 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.flatten_shape, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=config.num_classes), 
        )
        
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.flatten_shape, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=config.num_classes),  
        )
        
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.flatten_shape, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=config.num_classes),  
        )
        
        self.fc7 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.flatten_shape, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=config.num_classes), 
        )
        
        self.fc8 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.flatten_shape, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=config.num_classes), 
        )
        
        self.fc9 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.flatten_shape, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=config.num_classes),  
        )
        
        self.fc10 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.flatten_shape, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=config.num_classes),  
        )
       
        self.fcs = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7, self.fc8, self.fc9, self.fc10]
        # self.fcs = [self.fc for i in range(config.num_attribs)]
    
    def forward(self, x):
        features = self.feature_extractor(x)
        flat_features = features.view(features.size(0), -1)
        if config.pretext_task == 'label_augmentation':
            task_preds = []
            
            for i in range(config.num_tasks):
                #task_preds.append(torch.squeeze(torch.argmax(self.fcs[i](flat_features), dim=1)))
                
                task_preds.append(self.fcs[i](flat_features))
            
            return task_preds
        elif config.pretext_task == 'rotation':
            return self.fc_rotation(flat_features)
        else:
            print("Unknown pre-text task..")


class SSL_Network(torch.nn.Module):
    def __init__(self):
        super(SSL_Network, self).__init__()
        
        self.model = Network().to(config.device)
        self.previous_model = None
        
    def forward(self, x):
        task_predictions = self.model(x)
        return task_predictions
    
    def init_loss(self):
        if config.pretext_task == 'rotation': return torch.nn.CrossEntropyLoss()
        elif config.pretext_task == 'label_augmentation': return MultiTaskLoss()
        
    def init_optimizer(self, lr=config.lr, w_d=config.w_d, mom=config.momentum):
        return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=w_d, momentum=mom)
            
    def init_previous_downstream_model(self, model):
        self.previous_model = model

    def fit(self, trainloader, valloader=False, num_epochs=config.num_epochs):
        self.model = self.model.train()
        self.model = self.model.to(config.device)
        d = {'loss': [], 'accuracy':[]}
        
        total_loss_1 = 0
        
        loss_function = self.init_loss()
        optimizer = self.init_optimizer()
        
        for epoch in range(1, num_epochs+1):
            print("epoch : ", epoch)
            running_loss, total_loss_1 = 0, 0
            with tqdm(total=len(trainloader)) as pbar:
                for batch_idx, batch in enumerate(trainloader):
                    images, targets = batch
                    images = images.to(config.device)
                    targets = targets.to(config.device).long()
                    
                    outputs = self.model(images)
                    if config.pretext_task == 'rotation':
                        
                        loss = loss_function(outputs, targets)
                        pbar.set_postfix(Epochs='{}'.format(epoch),
                                     Loss='{0:6f}'.format(loss.item()))
                    elif config.pretext_task == 'label_augmentation':
                        individual_loss, reg_loss, loss = loss_function(outputs, targets, self.model, self.previous_model)
                        pbar.set_postfix(Epochs='{}'.format(epoch),
                                     Reg_loss='{0:4f}'.format(reg_loss),
                                     Loss='{0:6f}'.format(loss.item()))
                                     

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    pbar.update(1)
                
            epoch_loss = running_loss/len(trainloader)
            print("\nepoch_loss:", epoch_loss, " 1st attribute:", total_loss_1/len(trainloader))
			
            d['loss'].append(epoch_loss)
            print("................................")
        return d
    
    def evaluate(self, testloader):
        self.model.eval()
        
        bceloss = self.init_loss()
        total_loss, acc, count = 0.0, 0.0, 0
        
        with tqdm(total=len(testloader)) as pbar:
            for batch_idx, batch in enumerate(testloader):
                images, targets = batch
                images = images.to(config.device)
                targets = targets.to(config.device).long()
                
                output = self.model(images)
                loss = bceloss(output, targets)
                
                output = torch.transpose(torch.stack([torch.reshape(p>=0.5, (config.batch_size,)) for p in output]), 0, 1).float()
                acc += torch.sum(output == targets)
                total_loss += loss.item()
                count += config.batch_size
                
                pbar.set_postfix(Loss='{0:10f}'.format(loss.item()),
                                 accuracy='{0:.4f}'.format(float(float(acc)/float(count))))
                pbar.update(1)
        print(" Total Training loss: {0:.4f}".format(total_loss),
              " Test Accuracy:{0:.4f}".format(float(float(acc)/float(len(testloader)*config.batch_size))))

    def save_model(self, model_fname=None):
        torch.save(self.model.feature_extractor.state_dict(), model_fname)
        print("\nModel successfully saved.")
        
    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join('saved_models', 'ssl_model_'+config.model+'.pt')))
        print("Saved model loaded..")

    def load_weights(self, dmodel):
        d_dict = dmodel.state_dict()
        d = copy.deepcopy(self.model.state_dict())

        d['feature_extractor.c1.weight'] = copy.deepcopy(d_dict['c1.weight'])
        d['feature_extractor.c2.weight'] = copy.deepcopy(d_dict['c2.weight'])
        d['feature_extractor.c3.weight'] = copy.deepcopy(d_dict['c3.weight'])

        if config.num_layers > 3:
            d['feature_extractor.c4.weight'] = copy.deepcopy(d_dict['c4.weight'])
            d['feature_extractor.c5.weight'] = copy.deepcopy(d_dict['c5.weight'])

        if config.num_layers > 5:
            d['feature_extractor.c6.weight'] = copy.deepcopy(d_dict['c6.weight'])
            d['feature_extractor.c7.weight'] = copy.deepcopy(d_dict['c7.weight'])
            d['feature_extractor.c8.weight'] = copy.deepcopy(d_dict['c8.weight'])
        
        self.model.load_state_dict(d)
        print("Weights loaded from previous iteration for SSL network.")
