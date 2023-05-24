import os
import copy
import utils
import torch
import datetime
import numpy as np
import sys, argparse
import random
import config

# Arguments
parser=argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default= 0, help='(default=%(default)d)')
parser.add_argument('--experiment',default='cifar-100', type=str,required=False, help='(default=%(default)s)')
parser.add_argument('--approach', default='OWM', type=str, required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs', default=config.nepochs, type=int, required=False, help='(default=%(default)d)') # # 25
parser.add_argument('--lr', default=0.08, type=float, required=False, help='(default=%(default)f)') # # 0.02
parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--ssl_epochs', type=int, default= config.num_epochs, help='(default=%(default)d)')
parser.add_argument('--gpu', type=int, default= config.gpu, help='(default=%(default)d)')
args = parser.parse_args()

config.num_epochs = args.ssl_epochs
config.gpu = args.gpu
print("ssl_epochs:", config.num_epochs, args.ssl_epochs)
print("gpu:", config.gpu, args.gpu)

from owm import OWM
import svhn as dataloader
from ssl_models import SSL_Network
from ssl_dataloader import load_dataset, DataLoader

if config.num_layers==3:
    from ssl_models import OWM_ConvNet_3 as OWM_ConvNet
    from cnn_owm import OWM_Net_3 as OWM_Net
elif config.num_layers==5:
    from ssl_models import OWM_ConvNet_5 as OWM_ConvNet
    from cnn_owm import OWM_Net_5 as OWM_Net
elif config.num_layers==8:
    from ssl_models import OWM_ConvNet_8 as OWM_ConvNet
    from cnn_owm import OWM_Net_8 as OWM_Net

config.num_epochs = args.ssl_epochs
config.gpu = args.gpu
print("ssl_epochs:", config.num_epochs, args.ssl_epochs)
print("gpu:", config.gpu, args.gpu)

print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':', getattr(args,arg))
print('='*100)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    print('[CUDA unavailable]')
    sys.exit()

""
def load_weights_from_pretext_model(model, saved_model):
   
   
    saved_dict = saved_model.state_dict()

    d = copy.deepcopy(model.state_dict())

    d['c1.weight'] = copy.deepcopy(saved_dict['model.feature_extractor.c1.weight'])
    d['c2.weight'] = copy.deepcopy(saved_dict['model.feature_extractor.c2.weight'])
    d['c3.weight'] = copy.deepcopy(saved_dict['model.feature_extractor.c3.weight'])

    if config.num_layers > 3:
        d['c4.weight'] = copy.deepcopy(saved_dict['model.feature_extractor.c4.weight'])
        d['c5.weight'] = copy.deepcopy(saved_dict['model.feature_extractor.c5.weight'])

    if config.num_layers > 5:
        d['c6.weight'] = copy.deepcopy(saved_dict['model.feature_extractor.c6.weight'])
        d['c7.weight'] = copy.deepcopy(saved_dict['model.feature_extractor.c7.weight'])
        d['c8.weight'] = copy.deepcopy(saved_dict['model.feature_extractor.c8.weight'])
    
    model.load_state_dict(d)

    return model


def load_previous_fc_weights(model, saved_model):

    saved_dict = saved_model.state_dict()

    d = copy.deepcopy(model.state_dict())
    d['fc1.weight'] = copy.deepcopy(saved_dict['fc1.weight'])
    d['fc2.weight'] = copy.deepcopy(saved_dict['fc2.weight'])
    d['fc3.weight'] = copy.deepcopy(saved_dict['fc3.weight'])
    model.load_state_dict(d)
    return model


########################################################################################################################
# # MAIN # #
# #######################################################################################################################

print("Layers: ", config.num_layers)

#  Loading data
print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=args.seed)
print('Input size =', inputsize, '\nTask info =', taskcla)

# Initializing OWM_NET and 
print('Inits...')
owm_model = OWM_Net(inputsize).to(device)
ssl_model = SSL_Network().to(device)

utils.print_model_report(owm_model)
owm_obj = OWM(owm_model, nepochs=args.nepochs, args=args)

print('-'*100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
all_std = []
for t, ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t+1, data[t]['name']))
    print('*'*100)

    xtrain = data[t]['train']['x']
    ytrain = data[t]['train']['y']
    xvalid = data[t]['test']['x']
    yvalid = data[t]['test']['y']
    
    print(xtrain.shape, ytrain.shape)
    print(np.unique(ytrain, return_counts=True))

    if config.ssl_task == True:
        #----------------------------------------------------------------------------------
        # Train pretext model
        
        if config.pretext_task == 'label_augmentation':
            xtrain_pt, ytrain_pt = random_labelling_pretext(np.transpose(xtrain.cpu().numpy(), (0,2,3,1)))
        else:
            print("Unknown pretext task")
        
        # print(torch.min(xtrain_pt[0]), torch.max(xtrain_pt[0]))

        trainset = load_dataset(xtrain_pt, ytrain_pt)
        trainloader = DataLoader(trainset, batch_size = config.batch_size, shuffle = True)

        # # Loading previous downstream weights for pretext model
        if t>0:
            down_model_fname = 'saved_models/downstream_l'+str(config.num_layers)+'_t'+str(t-1)+'.pt'
            prev_downstream_model = OWM_Net(inputsize).to(device)
            prev_downstream_model.load_state_dict(torch.load(down_model_fname, map_location=config.device))
            ssl_model.load_weights(prev_downstream_model)
            ssl_model.init_previous_downstream_model(prev_downstream_model)

        # # Training pretext model here  
        if not os.path.exists('pretext_models'):
            os.mkdir('pretext_models')
        ssl_model.fit(trainloader)
        pretext_model_fname = 'pretext_models/feature_extractor_l'+str(config.num_layers)+'_t'+str(t)+'.pt'
        ssl_model.save_model(pretext_model_fname)
        print('-'*100)

        # # Loading pretext weights (conv) for owm downstream model        
        # owm_model = load_weights_from_pretext_model(owm_model, pretext_model_fname)     
        owm_model = load_weights_from_pretext_model(copy.deepcopy(owm_model), ssl_model)     
        
        # # Loading FC weights from last iteration's downstream model 
        if t >0:
            owm_model = load_previous_fc_weights(copy.deepcopy(owm_model), prev_downstream_model)
        
        owm_obj.update_model(copy.deepcopy(owm_model))
        
    xtrain = xtrain.to(device)
    ytrain = ytrain.to(device)
    xvalid = xvalid.to(device)
    yvalid = yvalid.to(device)

    # # # Training OWM
    best_std = owm_obj.train(t, xtrain, ytrain, xvalid, yvalid, data)
    all_std.append(best_std)
    owm_obj.save_model('saved_models/downstream_seed'+str(args.seed)+'_task'+str(t)+'.pt')
    # appr.save_model('saved_models/downstream_seed'+str(args.seed)+'_task'+str(t)+'.pt')
    print('-'*100)

    #----------------------------------------------------------------------------------
    # Test for all tasks
    for u in range(t+1):
        xtest = data[u]['test']['x'].to(device)
        ytest = data[u]['test']['y'].to(device)
        test_loss, test_acc = owm_obj.eval(xtest, ytest, best_std)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100*test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    xtest = data[5]['test']['x'].to(device)
    ytest = data[5]['test']['y'].to(device)
    test_loss, test_acc = owm_obj.eval(xtest, ytest, best_std)
    print('>>> Test on all tasks {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(data[5]['name'], test_loss, 100*test_acc))

    #quit()

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.2f}% '.format(100*acc[i, j]),end='')
    print()
print('*'*100)
print('ALL STD:', all_std)
print('Done!')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)

print("SVHN 5 task, SSL+ACL+OWM, ssl_epochs=", config.num_epochs)
print("args.seed:", args.seed)
print("Done!")
