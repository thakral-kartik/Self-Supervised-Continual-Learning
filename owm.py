import os
import sys, time
import numpy as np
import torch
import os
import utils
import config

# #######################################################################################################################
dtype = torch.float  # run on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class OWM(object):

    def __init__(self, model, nepochs=0, sbatch=64, clipgrad=10, args=None):
        self.model = model

        self.nepochs = nepochs
        self.sbatch = sbatch
        # self.lr = lr
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()

        self.Pc1 = torch.eye(3 * 3 * 3, device=device, dtype=dtype)
        self.Pc2 = torch.eye(64 * 3 * 3, device=device, dtype=dtype)
        self.Pc3 = torch.eye(128 * 3 * 3, device=device, dtype=dtype)
        if config.num_layers == 3:
            self.P1 = torch.eye(256 * 4 * 4, device=device, dtype=dtype)

        if config.num_layers == 5:
            self.Pc4 = torch.eye(256 * 2 * 2, device=device, dtype=dtype)
            self.Pc5 = torch.eye(512 * 2 * 2, device=device, dtype=dtype)
            self.P1 = torch.eye(1024 * 6 * 6, device=device, dtype=dtype)

        elif config.num_layers == 8:
            self.Pc4 = torch.eye(256 * 2 * 2, device=device, dtype=dtype)
            self.Pc5 = torch.eye(256 * 2 * 2, device=device, dtype=dtype)
            self.Pc6 = torch.eye(512 * 2 * 2, device=device, dtype=dtype)
            self.Pc7 = torch.eye(512 * 2 * 2, device=device, dtype=dtype)
            self.Pc8 = torch.eye(1024 * 2 * 2, device=device, dtype=dtype)
            self.P1 = torch.eye(1024 * 4 * 4, device=device, dtype=dtype)
        
        self.P2 = torch.eye(1000, device=device, dtype=dtype)
        self.P3 = torch.eye(1000, device=device, dtype=dtype)
        self.test_max = 0

        return

    def _get_optimizer(self, t, lr):
        
        lr_owm = lr
        fc1_params = list(map(id, self.model.fc1.parameters()))
        fc2_params = list(map(id, self.model.fc2.parameters()))
        fc3_params = list(map(id, self.model.fc3.parameters()))
        base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params,
                             self.model.parameters())
        optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': self.model.fc1.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc2.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc3.parameters(), 'lr': lr_owm}
                                     ], lr=lr, momentum=0.9)
        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data):
        best_loss = np.inf
        best_acc = 0
        best_std = 1 
        best_model = utils.get_model(self.model)
        # lr = self.lr
        # patience = self.lr_patience
        self.optimizer = self._get_optimizer(t, lr=config.owm_lrs[t])
        utils.print_optimizer_config(self.optimizer)
        nepochs = self.nepochs
        self.test_max = 0
        # Loop epochs
        try:
            for e in range(nepochs[t]):
                # Train
                if e == 0:
                    std =  0.9
                    print('CL check', std)
                if e != 0 and e % 10 == 0:
                    std /= 0.95
                    print('CL check', std)

                self.train_epoch(xtrain, ytrain, std, cur_epoch=e, nepoch=nepochs[t])
                train_loss, train_acc = self.eval(xtrain, ytrain, std)
                print('| [{:d}/5], Epoch {:d}/{:d}, | Train: loss={:.3f}, acc={:2.2f}% |'.format(t + 1, e + 1,
                                                                                                 nepochs[t], train_loss,
                                                                                                 100 * train_acc), end='')
                # # Valid: This is also the test data- but class-specific
                valid_loss, valid_acc = self.eval(xvalid, yvalid, std)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss, 100 * valid_acc), end='')
                print()

                # The entire test data
                xtest = data[5]['test']['x'].to(device)
                ytest = data[5]['test']['y'].to(device)

                _, test_acc = self.eval(xtest, ytest, std)

                if test_acc>self.test_max:                          # # Are they using test data here? (var: data) Yup. The entire test data, infact.
                    self.test_max = max(self.test_max, test_acc)
                    best_model = utils.get_model(self.model)
                    best_std = std

                print('>>> Test on All Task:->>> Max_acc : {:2.2f}%  Curr_acc : {:2.2f}%<<<'.format(100 * self.test_max, 100 * test_acc))

        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model, best_model)
        return best_std

    def train_epoch(self, x, y, std, cur_epoch=0, nepoch=0):
        self.model.train()
        
        r_len = np.arange(x.size(0))
        np.random.shuffle(r_len)
        r_len = torch.LongTensor(r_len).to(device)
        
        # Loop batches
        for i_batch in range(0, len(r_len), self.sbatch):
            b = r_len[i_batch:min(i_batch + self.sbatch, len(r_len))]
            images = x[b]
            targets = y[b]

            # Forward
            
            output, h_list, x_list = self.model.forward(images, std)
            loss = self.ce(output, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            lamda = i_batch / len(r_len)/nepoch + cur_epoch/nepoch      # # what is happening here?
            # print("i_batch: ", i_batch, " r_len: ", r_len, " cur_epoch: ", cur_epoch, " nepoch: ", nepoch, "Lambda: ", lamda)

            # # being used in weight modifications of conv and fc layers
            alpha_array = [1.0 * 0.00001 ** lamda, 1.0 * 0.0001 ** lamda, 1.0 * 0.01 ** lamda, 1.0 * 0.1 ** lamda]

            def pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):

                with torch.no_grad():
                    if cnn:   # for conv layers
                        _, _, H, W = x.shape
                        F, _, HH, WW = w.shape
                        S = stride  # stride
                        Ho = int(1 + (H - HH) / S)
                        Wo = int(1 + (W - WW) / S)
                        for i in range(Ho):
                            for j in range(Wo):
                                # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                                r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                                # r = r[:, range(r.shape[1] - 1, -1, -1)]
                                k = torch.mm(p, torch.t(r))
                                p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                        w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
                    
                    else:       # for fc layers
                        r = x
                        k = torch.mm(p, torch.t(r))
                        p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                        w.grad.data = torch.mm(w.grad.data, torch.t(p.data))
                
            for n, w in self.model.named_parameters():  # # since we already know our CNN model
                if n == 'c1.weight':
                    pro_weight(self.Pc1, x_list[0], w, alpha=alpha_array[0], stride=2)

                if n == 'c2.weight':
                    pro_weight(self.Pc2, x_list[1], w, alpha=alpha_array[0], stride=2)

                if n == 'c3.weight':
                    pro_weight(self.Pc3, x_list[2], w, alpha=alpha_array[0], stride=2)

                if config.num_layers > 3:
                    if n == 'c4.weight':
                        pro_weight(self.Pc4, x_list[3], w, alpha=alpha_array[0], stride=2)
                
                    if n == 'c5.weight':
                        pro_weight(self.Pc5, x_list[4], w, alpha=alpha_array[0], stride=2)

                if config.num_layers > 5:
                    if n == 'c6.weight':
                        pro_weight(self.Pc6, x_list[5], w, alpha=alpha_array[0], stride=2)

                    if n == 'c7.weight':
                        pro_weight(self.Pc7, x_list[6], w, alpha=alpha_array[0], stride=2)

                    if n == 'c8.weight':
                        pro_weight(self.Pc8, x_list[7], w, alpha=alpha_array[0], stride=2)

                if n == 'fc1.weight':
                    pro_weight(self.P1,  h_list[0], w, alpha=alpha_array[1], cnn=False)

                if n == 'fc2.weight':
                    pro_weight(self.P2,  h_list[1], w, alpha=alpha_array[2], cnn=False)

                if n == 'fc3.weight':
                    pro_weight(self.P3,  h_list[2], w, alpha=alpha_array[3], cnn=False)

            # Apply step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        # return

    def eval(self, x, y, std):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        r = np.arange(x.size(0))
        r = torch.LongTensor(r).to(device)
        with torch.no_grad():
            # Loop batches
            for i in range(0, len(r), self.sbatch):
                b = r[i:min(i + self.sbatch, len(r))]
                images = x[b]
                targets = y[b]

                # Forward
                output,  _, _ = self.model.forward(images, std)
                loss = self.ce(output, targets)
                _, pred = output.max(1)
                hits = (pred % 10 == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy().item() * len(b)
                total_acc += hits.sum().data.cpu().numpy().item()
                total_num += len(b)

        return total_loss / total_num, total_acc / total_num

    def save_model(self, model_fname=None):
        if not os.path.isdir('saved_models'):
            os.mkdir('saved_models')
        torch.save(self.model.state_dict(), model_fname)
        print("\nModel successfully saved.")

    def update_model(self, model):
        self.model = model
