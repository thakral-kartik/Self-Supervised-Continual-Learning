# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from torchsummary import summary
import skimage.io as io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OWM_Net_3(torch.nn.Module):

    def __init__(self, inputsize):
        super(OWM_Net_3, self).__init__()
        ncha, size, _ = inputsize
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(256)

        self.padding = torch.nn.ReplicationPad2d(1)

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False)
        
        self.fc1 = torch.nn.Linear(256 * 4 * 4, 1000, bias=False)         
        self.fc2 = torch.nn.Linear(1000, 1000, bias=False)
        self.fc3 = torch.nn.Linear(1000, 10,  bias=False)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

        return
    

    def forward(self, x):
        h_list = []
        x_list = []
            
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
        
        
        con3_p = self.maxpool(con3) #64*256*9*9
         
        h = con3_p.view(x.size(0), -1) #64*4096
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))
        y = self.fc3(h)
        return y, h_list, x_list

# -------------------------------------------------------------------------------------------------#

class OWM_Net_5(torch.nn.Module):

    def __init__(self, inputsize):
        super(OWM_Net_5, self).__init__()
        ncha, size, _ = inputsize
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=2, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=False)
        self.c4 = torch.nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0, bias=False)
        self.c5 = torch.nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=0, bias=False)

        self.fc1 = torch.nn.Linear(1024 * 6 * 6, 1000, bias=False)       
        self.fc2 = torch.nn.Linear(1000, 1000, bias=False)
        self.fc3 = torch.nn.Linear(1000, 10,  bias=False)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

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

        con3_p = self.padding(con3_p)
        x_list.append(torch.mean(con3_p, 0, True))
        con4 = self.drop1(self.relu(self.c4(con3_p)))

        con4_p = con4

        con4_p = self.padding(con4_p)
        x_list.append(torch.mean(con4_p, 0, True))
        con5 = self.drop1(self.relu(self.c5(con4_p)))
        con5_p = con5

        h = con5_p.view(x.size(0), -1)
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))
        y = self.fc3(h)
        return y, h_list, x_list

# -------------------------------------------------------------------------------------------------#

class OWM_Net_8(torch.nn.Module):

    def __init__(self, inputsize):
        super(OWM_Net_8, self).__init__()
        ncha, size, _ = inputsize
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)

        self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=2, stride=1, padding=0, bias=False)
        self.c2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1,  padding=0, bias=False)
        self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=False)
        self.c4 = torch.nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0, bias=False)
        self.c5 = torch.nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0, bias=False)
        self.c6 = torch.nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0, bias=False)
        self.c7 = torch.nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=0, bias=False)
        self.c8 = torch.nn.Conv2d(1024, 1024, kernel_size=2, stride=1, padding=0, bias=False)

        self.fc1 = torch.nn.Linear(1024 * 4 * 4, 1000, bias=False)         # # update this
        self.fc2 = torch.nn.Linear(1000, 1000, bias=False)
        self.fc3 = torch.nn.Linear(1000, 10,  bias=False)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

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
        con2_p = con2

        con2_p = self.padding(con2_p)
        x_list.append(torch.mean(con2_p, 0, True))
        con3 = self.drop1(self.relu(self.c3(con2_p)))
        con3_p = self.maxpool(con3)      

        con3_p = self.padding(con3_p)
        x_list.append(torch.mean(con3_p, 0, True))
        con4 = self.drop1(self.relu(self.c4(con3_p)))
        con4_p = con4

        con4_p = self.padding(con4_p)
        x_list.append(torch.mean(con4_p, 0, True))
        con5 = self.drop1(self.relu(self.c5(con4_p)))
        con5_p = self.maxpool(con5)
        con5_p = con5_p

        # # adding from here
        con5_p = self.padding(con5_p)
        x_list.append(torch.mean(con5_p, 0, True))
        con6 = self.drop1(self.relu(self.c6(con5_p)))
        con6_p = con6

        con6_p = self.padding(con6_p)
        x_list.append(torch.mean(con6_p, 0, True))
        con7 = self.drop1(self.relu(self.c7(con6_p)))
        con7_p = self.maxpool(con7)  # # removed maxpool

        con7_p = self.padding(con7_p)
        x_list.append(torch.mean(con7_p, 0, True))
        con8 = self.drop1(self.relu(self.c8(con7_p)))
        con8_p = con8

       

        h = con8_p.view(x.size(0), -1)
        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc1(h))

        h_list.append(torch.mean(h, 0, True))
        h = self.relu(self.fc2(h))

        h_list.append(torch.mean(h, 0, True))
        y = self.fc3(h)
        return y, h_list, x_list

# -------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    x = Net((3,32,32)).cuda()
    d = x.state_dict()
    print(d.keys())

    summary(x, (3, 32, 32))

    print(d['c1.weight'].shape)
    print(d['c2.weight'].shape)
    print(d['c3.weight'].shape)
    print(d['c4.weight'].shape)
    print(d['c5.weight'].shape)
    print(d['c6.weight'].shape)
    print(d['c7.weight'].shape)
    print(d['c8.weight'].shape)
    # print(d['c9.weight'].shape)
    # print(d['c10.weight'].shape)
    # return d
