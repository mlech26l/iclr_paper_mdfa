import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide tensorflow logs
import tensorflow as tf
import fa_model
import fa_convnet
import argparse
from tqdm import tqdm
from datetime import datetime
import os
from sklearn.utils import shuffle

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

def select_dataset(dataset,convnet):
    if(dataset == 'mnist'):
        mnist = tf.keras.datasets.mnist
        (train_x, train_y),(test_x, test_y) = mnist.load_data()
        train_x, test_x = train_x/255.0, test_x/255.0
        if(convnet):
            train_x = train_x.reshape([-1,28,28,1])
            test_x = test_x.reshape([-1,28,28,1])
        else:
            train_x = train_x.reshape([-1,28*28])
            test_x = test_x.reshape([-1,28*28])
    elif(dataset == 'cifar10'):
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
        train_x, test_x = train_x/255.0, test_x/255.0
        if(convnet):
            train_x = train_x.reshape([-1,32,32,3])
            test_x = test_x.reshape([-1,32,32,3])
        else:
            train_x = train_x.reshape([-1,32*32*3])
            test_x = test_x.reshape([-1,32*32*3])
        train_y = train_y.reshape([-1])
        test_y = test_y.reshape([-1])
    elif(dataset == 'cifar100'):
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()
        train_x, test_x = train_x/255.0, test_x/255.0
        if(convnet):
            train_x = train_x.reshape([-1,32,32,3])
            test_x = test_x.reshape([-1,32,32,3])
        else:
            train_x = train_x.reshape([-1,32*32*3])
            test_x = test_x.reshape([-1,32*32*3])
        train_y = train_y.reshape([-1])
        test_y = test_y.reshape([-1])
    else:
        raise ValueError("Unknown dataset "+str(dataset))

    return train_x,train_y,test_x,test_y

def split_validation(seed,train_x,train_y):
    perm = np.random.RandomState(args.seed).permutation(train_x.shape[0])
    train_x = train_x[perm]
    train_y = train_y[perm]
    valid_size = int(0.15*train_x.shape[0])
    valid_x,valid_y = train_x[:valid_size],train_y[:valid_size]
    train_x,train_y = train_x[valid_size:],train_y[valid_size:]
    return train_x,train_y,valid_x,valid_y

def shrink_dimension(x,y,num_classes,seed):
    max_classes = np.max(y)+1
    if(num_classes <= 1 or num_classes > max_classes):
        return x,y

    rng = np.random.RandomState(seed)
    perm = rng.permutation(max_classes)[:num_classes]

    all_x = []
    all_y = []

    for i in range(num_classes):
        select_x = x[y==perm[i]]

        select_y = np.ones([select_x.shape[0]],dtype=np.int32)*i
        all_x.append(select_x)
        all_y.append(select_y)

    all_x = np.concatenate(all_x,axis=0)
    all_y = np.concatenate(all_y,axis=0)

    perm = rng.permutation(all_x.shape[0])
    return all_x[perm],all_y[perm]

# Parse arugments
parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--max_epochs',  type=int, default=100)
parser.add_argument('--model',default='bp')
parser.add_argument('--convnet',action="store_true")
parser.add_argument('--activation',default='tanh')
parser.add_argument('--forward_init',default='U1')
parser.add_argument('--backward_init',default='U1')
parser.add_argument('--optimizer',default='rmsprop')
parser.add_argument('--classes',default=-1,type=int)
parser.add_argument('--seed',default=0,type=int)
parser.add_argument('--batch_size',default=64,type=int)
parser.add_argument('--lr',  type=float, default=0.0001)
parser.add_argument('--l2',  type=float, default=0.0)
parser.add_argument('--dataset', default='mnist')

args = parser.parse_args()
assert args.seed >= 0 and args.seed < 10
assert args.classes >= 0
if(args.model == "mdfa"):
    args.forward_init = args.forward_init.replace("U","P").replace("N","A")
    args.backward_init = args.backward_init.replace("U","P").replace("N","A")


activation_name = args.activation
if(args.convnet):
    activation_name = "cnn_"+activation_name

base_dir = os.path.join("results",args.dataset,activation_name,"seed_{:d}".format(args.seed))
if(not os.path.exists(base_dir)):
    os.makedirs(base_dir)

hashed = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(base_dir, "{}_c{:04d}_{}.csv".format(
    args.model,args.classes,hashed
))

train_x,train_y,test_x,test_y = select_dataset(args.dataset,args.convnet)

train_x, train_y = shrink_dimension(train_x,train_y,args.classes,args.seed)
test_x, test_y = shrink_dimension(test_x,test_y,args.classes,args.seed)
train_x,train_y,valid_x,valid_y = split_validation(args.seed,train_x,train_y)

output_dim = args.classes
if(output_dim == 2):
    output_dim = 1

if(args.dataset == "mnist"):
    layers = [200,200]
else:
    layers = [1024,1024]
convnet_layers = [(96,5,2),(96,3,2),(96,3,1)]

nn_type = select_model(args.model,args.convnet)
if(args.convnet):
    nn = nn_type(
        input_dim = train_x.shape[1],
        input_channels = train_x.shape[3], 
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
        input_dim=test_x.shape[1],
        output_dim=output_dim,
        layer_dims = layers,
        activation = args.activation,
        learning_rate = args.lr,
        l2_coeff=args.l2,
        forward_init = args.forward_init,
        backward_init = args.backward_init,
        optimizer = args.optimizer,
    )

print("train x shape: ",str(train_x.shape))
print("train y shape: ",str(train_y.shape))
print("test x shape: ",str(test_x.shape))
print("test y shape: ",str(test_y.shape))
print('Training started')

# Log optimizer (train error)
best_train_epoch = [1000,0,0] # train_loss,train_acc, valid_ac
# Log generalization (valid->test error)
best_valid_acc = -1

for e in range(args.max_epochs):
    valid_loss = []
    valid_accuracy = []
    valid_batch_size = 500
    if(valid_x.shape[0] < 1000):
        valid_batch_size = valid_x.shape[0]
    for i in range(valid_x.shape[0]//valid_batch_size):
        loss,acc = nn.evaluate(valid_x[i*valid_batch_size:(i+1)*valid_batch_size],valid_y[i*valid_batch_size:(i+1)*valid_batch_size])
        valid_loss.append(loss)
        valid_accuracy.append(acc)
    valid_loss = np.mean(valid_loss)
    valid_accuracy = np.mean(valid_accuracy)

    if(valid_accuracy > best_valid_acc):
        #  New best 
        nn.save("temp_checkpoint")
        best_valid_acc = valid_accuracy

    train_x, train_y = shuffle(train_x, train_y)

    train_loss = []
    train_accucacy = []
    iters = train_x.shape[0]//args.batch_size
    for i in range(iters):
        indx = np.arange(i*args.batch_size,(i+1)*args.batch_size)
        loss,acc = nn.train(train_x[indx],train_y[indx])
        train_loss.append(loss)
        train_accucacy.append(acc)

    train_loss = np.mean(train_loss)
    train_accucacy = np.mean(train_accucacy)
    
    if(train_accucacy > best_train_epoch[1]):
        best_train_epoch = [train_loss,train_accucacy,valid_accuracy]

    print('Trained {}/{} epochs, train_loss {:0.2f}, train_acc: {:0.2f} val_loss: {:0.2f} val_acc: {:0.2f}'.format(e+1,args.max_epochs,train_loss,100*train_accucacy,valid_loss,100*valid_accuracy))

nn.load("temp_checkpoint")
test_loss = []
test_accuracy = []
valid_batch_size = 100
if(test_x.shape[0] < 1000):
    valid_batch_size = test_x.shape[0]
for i in range(test_x.shape[0]//valid_batch_size):
    loss,acc = nn.evaluate(test_x[i*valid_batch_size:(i+1)*valid_batch_size],test_y[i*valid_batch_size:(i+1)*valid_batch_size])
    test_loss.append(loss)
    test_accuracy.append(acc)
test_loss = np.mean(test_loss)
test_accuracy = np.mean(test_accuracy)

all_results = [best_train_epoch[0],best_train_epoch[1],best_train_epoch[2],best_valid_acc,test_accuracy]
np.savetxt(result_file,np.array(all_results))

print("Best train loss: {:0.2f} ({:0.2f}% acc), best valid acc: {:0.2f}%, test acc: {:0.2f}%".format(
    best_train_epoch[0],
    100*best_train_epoch[1],
    100*best_valid_acc,
    100*test_accuracy
))

with open("hyperparam_log.txt","a") as f:
    f.write("{} on {}[{}]: {:0.2f}% valid acc. args: {}\n".format(
        args.model,args.dataset,args.classes,best_valid_acc*100,str(args)
    ))