import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide tensorflow logs
import tensorflow as tf
import fa_model
import fa_convnet
from imagenet_iterator import ImageNetIterator
from inmemory_imagenet_iterator import InMemoryImageNetIterator
import argparse
from tqdm import tqdm
import time
from datetime import datetime

def select_model(model,convnet):
    if(convnet):
        if(model == 'bp'):
            nn_type = fa_convnet.Convnet
        elif(model == 'fa'):
            nn_type = fa_convnet.FAConvnet
        elif(model == 'dfa'):
            nn_type = fa_convnet.DFAConvnet
        elif(model == 'mdfa'):
            nn_type = fa_convnet.mDFAConvnet
        else:
            raise ValueError("Unknown model '{}'".format(model))
    else:
        if(model == 'bp'):
            nn_type = fa_model.Net
        elif(model == 'fa'):
            nn_type = fa_model.FAnet
        elif(model == 'dfa'):
            nn_type = fa_model.DFAnet
        elif(model == 'mdfa'):
            nn_type = fa_model.mDFAnet
        else:
            raise ValueError("Unknown model '{}'".format(model))
    return nn_type

# Parse arugments
parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--max_epochs',  type=int, default=100)
parser.add_argument('--model',default='bp')
parser.add_argument('--convnet',action="store_true")
parser.add_argument('--activation',default='tanh')
parser.add_argument('--optimizer',default='rmsprop')
parser.add_argument('--forward_init',default='U1')
parser.add_argument('--backward_init',default='U1')
parser.add_argument('--classes',default=-1,type=int)
parser.add_argument('--seed',default=0,type=int)
parser.add_argument('--batch_size',default=64,type=int)
parser.add_argument('--lr',  type=float, default=0.0001)
parser.add_argument('--l2',  type=float, default=0.0)

args = parser.parse_args()
assert not (args.large_cnn and args.convnet)
assert args.seed >= 0 and args.seed < 10
assert args.classes >= 0
activation_name = args.activation

if(args.convnet):
    activation_name = "cnn_"+activation_name

base_dir = os.path.join("results","imagenet",activation_name,"seed_{:d}".format(args.seed))
if(not os.path.exists(base_dir)):
    os.makedirs(base_dir)

hashed = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(base_dir, "{}_c{:04d}_{}.csv".format(
    args.model,args.classes,hashed
))


output_dim = args.classes
if(output_dim == 2):
    output_dim = 1

# For performance reasons, if we only train on a few classes we can load all images into the main memory
if(args.classes <= 9):
    # The images of 50 classes require ~10GB of memory to store, and more than double of it for loading
    train_iter = InMemoryImageNetIterator("training",num_classes=args.classes,seed=args.seed,flatten=not args.convnet)
    valid_iter = InMemoryImageNetIterator("validation",num_classes=args.classes,seed=args.seed,flatten=not args.convnet)
else:
    train_iter = ImageNetIterator("training",num_classes=args.classes,seed=args.seed,flatten=not args.convnet)
    valid_iter = ImageNetIterator("validation",num_classes=args.classes,seed=args.seed,flatten=not args.convnet)

layers = [1024,1024,1024,1024]
convnet_layers = [(96,9,4),(96,3,2),(128,5,1),(128,3,2),(192,3,1),(192,3,2),(384,3,1)]

nn_type = select_model(args.model,args.convnet)
if(args.convnet):
    nn = nn_type(
        input_dim = 224,
        input_channels = 3, 
        output_dim = output_dim, 
        layer_dims=convnet_layers,
        activation = args.activation,
        learning_rate = args.lr,
        l2_coeff=args.l2,
        forward_init = args.forward_init,
        backward_init = args.backward_init,
        optimizer = args.optimizer,

    )
else:
    nn = nn_type(
        input_dim=224*224*3,
        output_dim=output_dim,
        layer_dims = layers,
        activation = args.activation,
        learning_rate = args.lr,
        l2_coeff=args.l2,
        forward_init = args.forward_init,
        backward_init = args.backward_init,
        optimizer = args.optimizer,
    )

print('Training started')

# Log optimizer (train error)
best_train_epoch = [1000,0,0] # train_loss,train_acc, valid_ac
# Log generalization (valid->test error)
best_valid_acc = [0,0]

for e in range(args.max_epochs):
    valid_top_k = []
    valid_accuracy = []
    valid_batch_size = 100
    for x,y in valid_iter.iterate(batch_size=100,shuffle=False):
        if(args.classes >= 5):
            loss,acc,top_k = nn.evaluate_top_k(x,y)
        else:
            loss,acc = nn.evaluate(x,y)
            top_k = 1
        valid_top_k.append(top_k)
        valid_accuracy.append(acc)
    valid_accuracy = np.mean(valid_accuracy)
    valid_top_k = np.mean(valid_top_k)
    if(valid_accuracy > best_valid_acc[0]):
        #  New best 
        best_valid_acc = [valid_accuracy,valid_top_k]

    train_loss = []
    train_accucacy = []
    epoch_start = time.time()
    for x,y in train_iter.iterate_with_prefetch(batch_size=args.batch_size,shuffle=True):
        loss,acc = nn.train(x,y)
        train_loss.append(loss)
        train_accucacy.append(acc)


    train_loss = np.mean(train_loss)
    train_accucacy = np.mean(train_accucacy)

    print('Trained {}/{} epochs ({:0.1f} img/s), train_loss {:0.2f}, train_acc: {:0.2f}  val_acc: {:0.2f} ({:0.2f}% top 5)'.format(e+1,args.max_epochs,train_iter.size()/(time.time()-epoch_start),train_loss,100*train_accucacy,100*valid_accuracy,100*valid_top_k))
    if(train_accucacy > best_train_epoch[1]):
        best_train_epoch = [train_loss,train_accucacy,valid_accuracy]

all_results = [best_train_epoch[0],best_train_epoch[1],best_train_epoch[2],best_valid_acc[0],best_valid_acc[1]]
np.savetxt(result_file,np.array(all_results))

with open("imagenet_hyperparameter_log.txt","a") as f:
    f.write("{} on imagenet[{}]: {:0.2f}% valid acc. args: {}\n".format(
        args.model,args.classes,best_valid_acc[0]*100,str(args)
    ))