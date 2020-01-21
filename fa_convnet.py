import numpy as np
import tensorflow as tf
import os
from init_parser import parse_initializer

class Convnet:

    def __init__(self,
        input_dim,
        input_channels, 
        output_dim, 
        layer_dims,
        activation="tanh",
        learning_rate=0.001,
        l2_coeff=0.0,
        forward_init = "U1",
        backward_init = "U1",
        optimizer = "rmsprop"
        ):
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.layer_dims = layer_dims
        self.activation = activation
        self.l2_coeff = l2_coeff
        self.forward_init = forward_init
        self.backward_init = backward_init


        self.x = tf.placeholder(dtype=tf.float32,shape=[None,self.input_dim,self.input_dim,self.input_channels])
        
        # Preprocess input
        x = self.build_preprocess(self.x)
        # Build model
        self.y = self.build_forward(x)

        # Build loss and training op
        self._target_y = tf.placeholder(dtype=tf.int32,shape=[None])

        if(self.output_dim==1):
            target_y = tf.cast(tf.reshape(self._target_y,shape=[-1,1]),dtype=tf.float32)
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_y,logits=self.y)
            self.loss = tf.reduce_mean(self.batch_loss)
            model_prediction = tf.reshape(tf.cast(tf.round(tf.nn.sigmoid(self.y)),dtype=tf.int32),shape=[-1])
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, self._target_y), dtype=tf.float32))
        else:
            self.batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._target_y,logits=self.y)
            self.loss = tf.reduce_mean(self.batch_loss)
            model_prediction = tf.cast(tf.argmax(input=self.y, axis=1),dtype=np.int32)
            # correct_label = tf.cast(tf.argmax(input=self._target_y, axis=1),dtype=np.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, self._target_y), dtype=tf.float32))

        self.top_k = None

        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))
        self.vars,self.grads = self.build_gradient()

        self._add_l2()


        if(optimizer == "rmsprop"):
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        elif(optimizer == "adam"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif(optimizer == "sgd"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError("Unknown optimizer '{}'".format(optimizer))

        self.train_step = optimizer.apply_gradients(zip(self.grads, self.vars))

        self.constraints = self.build_constrain_op()

        # HACK: https://github.com/tensorflow/tensorflow/issues/24828
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _add_l2(self):
        if(self.l2_coeff <= 0.0):
            return
        new_grads = []
        for i in range(len(self.vars)):
            new_grads.append(self.grads[i] + self.l2_coeff*self.vars[i])

        self.grads = new_grads

    def _add_top_k_accuracy(self,k):
        correct_k = tf.nn.in_top_k(
            predictions=self.y,
            targets=self._target_y,
            k=k)
        correct_k = tf.cast(correct_k,dtype=tf.float32)
        self.top_k =  tf.reduce_mean(correct_k)

    def count_fanin(self,shape):
        fanin = 1
        for i in range(len(shape)-1):
            fanin *= int(shape[i])
        return fanin

    def _forward_weight(self,shape,name,first_layer=False):
        init = self.forward_init
        if(first_layer):
            # HACK: First layer can always be +-
            init = init.replace("P","U").replace("A","N")

        fanin_init = 1.0/np.sqrt(self.count_fanin(shape))
        initializer = parse_initializer(init,fanin_init)
        return tf.get_variable(
            name=name,
            shape=shape,
            dtype=tf.float32,
            initializer=initializer
        )

    def _backward_weight(self,shape,name,transpose=False):
        fanin_init = 1.0/np.sqrt(self.count_fanin(shape))
        initializer = parse_initializer(self.backward_init,fanin_init)

        # transpose
        if(transpose):
            shape = list(shape)[::-1]

        return tf.get_variable(
            name=name,
            shape=shape,
            dtype=tf.float32,
            trainable = False,
            initializer=initializer
        )
    def _get_bias(self,size,name):
        bias_const_init = 0
        if(self.activation == "relu"):
            bias_const_init = 0.1

        return tf.get_variable(
            name=name,
            shape=[size],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))


    def build_constrain_op(self):
        return None

    def build_preprocess(self,x):
        return x

    def build_forward(self,x):
        self._tf_layers = []
        self._tf_layers_output = []
        self._tf_layers_input = []
        self._tf_layers_preactivation = []

        head = x
        for i in range(len(self.layer_dims)):
            self._tf_layers_input.append(head)

            in_size = int(head.shape[1])
            in_channels = int(head.shape[3])

            filters, kernel_size,stride = self.layer_dims[i]

            w = self._forward_weight([kernel_size,kernel_size,in_channels,filters],"W_{}".format(i),first_layer=i==0)
            b = self._get_bias(filters,"b_{}".format(i))
            print("conv Layer {}, kernel: [{}], bias: {}".format(i,str(w.shape),str(b.shape)))

            a = tf.nn.conv2d(head,filter=w,strides=[1,stride,stride,1],padding="VALID")+b

            print("pre Head shape: {}".format(str(head.shape)))
            self._tf_layers_preactivation.append(a)
            head = self.activation_function(a)
            print("post Head shape: {}".format(str(head.shape)))
            self._tf_layers.append((w,b))
            self._tf_layers_output.append(head)

        total_size = int(head.shape[1])*int(head.shape[2])*int(head.shape[3])
        head = tf.reshape(head,shape=[-1,total_size])
        self._tf_layers_input.append(head)
        w = self._forward_weight([int(head.shape[1]),self.output_dim],"W_out")
        b = self._get_bias(self.output_dim,"b_out")
        y = tf.matmul(head,w) + b
        self._tf_layers_preactivation.append(y)
        self._tf_layers.append((w,b))
        self._tf_layers_output.append(y)

        print("y shape: ",str(y.shape))
        return y

    def build_gradient(self):
        logit_grad = tf.gradients(self.batch_loss,self.y)[0]
        print("logit_grad shape: ",str(logit_grad.shape))
        vars = []
        grads = []

        back_prob = logit_grad
        self._B = []

        # Fully-connected logit layer
        w,b = self._tf_layers[-1]
        y = self._tf_layers_output[-1]
        x = self._tf_layers_input[-1]
        a = self._tf_layers_preactivation[-1]
        d_z = back_prob
        w_grad = tf.matmul(tf.transpose(x), d_z)
        x_grad = tf.matmul(d_z,tf.transpose(w))
        b_grad = tf.reduce_sum(d_z,axis=0)

        vars.append(w)
        grads.append(w_grad)
        vars.append(b)
        grads.append(b_grad)

        back_prob = tf.reshape(x_grad,shape=tf.shape(self._tf_layers_output[-2]))

        # Convolutional layers
        for i in reversed(range(len(self._tf_layers)-1)):
            w,b = self._tf_layers[i]
            y = self._tf_layers_output[i]
            x = self._tf_layers_input[i]
            a = self._tf_layers_preactivation[i]
            filters, kernel_size,stride = self.layer_dims[i]

            print("back_prop shape",str(back_prob.shape))
            print("y shape",str(y.shape))
            print("a shape",str(a.shape))
            print("x shape",str(x.shape))

            d_z = tf.multiply(back_prob, self.grad_activation(a))
            print("d_z shape: ",str(d_z.shape))
            w_grad = tf.nn.conv2d_backprop_filter(x,w.shape,out_backprop=d_z,strides=[1,stride,stride,1],padding="VALID")
            print("w_grad shape: ",str(w_grad.shape))

            back_prob = tf.nn.conv2d_backprop_input(input_sizes=tf.shape(x),filter=w,out_backprop=d_z,strides=[1,stride,stride,1],padding="VALID")

            # Reduce batch, height and width dimension
            b_grad = tf.reduce_sum(d_z,axis=[0,1,2])
            print("b_grad shape: ",str(b_grad.shape))

            vars.append(w)
            grads.append(w_grad)
            vars.append(b)
            grads.append(b_grad)

        return vars,grads

    def grad_activation(self,x):
        if(self.activation == "tanh"):
            return 1.0-tf.square(tf.nn.tanh(x))
        elif(self.activation == "relu"):
            return tf.where(x >= 0, tf.ones_like(x,dtype=tf.float32), tf.zeros_like(x,dtype=tf.float32))
        elif(self.activation == "pl"):
            return tf.where((x >= -1) & (x<= 1), tf.ones_like(x,dtype=tf.float32), tf.zeros_like(x,dtype=tf.float32))
        else:
            raise ValueError("Unknown activation function")

    def activation_function(self,x):
        if(self.activation == "tanh"):
            return tf.nn.tanh(x)
        elif(self.activation == "relu"):
            return tf.nn.relu(x)
        elif(self.activation == "pl"):
            x = tf.maximum(x,-1)
            x = tf.minimum(x,1)
            return x
        else:
            raise ValueError("Unknown activation function")

    def forward(self,x):
        feed_dict = {self.x: x}
        
        return self.sess.run(self.y,feed_dict)

    def evaluate(self,x,y):
        feed_dict = {self.x: x,self._target_y:y}
        
        loss,accuracy = self.sess.run([self.loss,self.accuracy],feed_dict)
        return loss,accuracy

    def evaluate_top_k(self,x,y,k=5):
        if(self.top_k is None):
            self._add_top_k_accuracy(k)

        feed_dict = {self.x: x,self._target_y:y}
        
        loss,accuracy,top_k = self.sess.run([self.loss,self.accuracy,self.top_k],feed_dict)
        return loss,accuracy,top_k
        
    def train(self,x,y):
        feed_dict = {self.x: x,self._target_y:y}
        
        loss,acc,_ = self.sess.run([self.loss,self.accuracy,self.train_step],feed_dict)
        if(not self.constraints is None):
            self.sess.run(self.constraints)
        return loss,acc

    def save(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, '-model')
        # Create a new saver object
        self.saver = tf.train.Saver()
        filename = self.saver.save(self.sess, checkpoint_path)

    def load(self,path):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(path,'-model'))



class FAConvnet(Convnet):


    def build_gradient(self):
        logit_grad = tf.gradients(self.batch_loss,self.y)[0]
        vars = []
        grads = []

        back_prob = logit_grad
        self._B = []

        # Fully-connected logit layer
        w,b = self._tf_layers[-1]
        y = self._tf_layers_output[-1]
        x = self._tf_layers_input[-1]
        a = self._tf_layers_preactivation[-1]
        d_z = back_prob
        w_grad = tf.matmul(tf.transpose(x), d_z)

        B_transposed = self._backward_weight(w.shape,name="B_out",transpose=True)
        self._B.append(B_transposed)

        x_grad = tf.matmul(d_z,B_transposed)
        b_grad = tf.reduce_sum(d_z,axis=0)

        vars.append(w)
        grads.append(w_grad)
        vars.append(b)
        grads.append(b_grad)

        back_prob = tf.reshape(x_grad,shape=tf.shape(self._tf_layers_output[-2]))

        # Convolutional layers
        for i in reversed(range(len(self._tf_layers)-1)):
            w,b = self._tf_layers[i]
            y = self._tf_layers_output[i]
            x = self._tf_layers_input[i]
            a = self._tf_layers_preactivation[i]
            filters, kernel_size,stride = self.layer_dims[i]

            print("back_prop shape",str(back_prob.shape))
            print("y shape",str(y.shape))
            print("a shape",str(a.shape))
            print("x shape",str(x.shape))

            d_z = tf.multiply(back_prob, self.grad_activation(a))
            print("d_z shape: ",str(d_z.shape))

            w_grad = tf.nn.conv2d_backprop_filter(x,w.shape,out_backprop=d_z,strides=[1,stride,stride,1],padding="VALID")
            print("w_grad shape: ",str(w_grad.shape))

            B = self._backward_weight(w.shape,name="B_{}".format(i))
            self._B.append(B)
            back_prob = tf.nn.conv2d_backprop_input(input_sizes=tf.shape(x),filter=B,out_backprop=d_z,strides=[1,stride,stride,1],padding="VALID")

            # Reduce batch, height and width dimension
            b_grad = tf.reduce_sum(d_z,axis=[0,1,2])
            print("b_grad shape: ",str(b_grad.shape))

            vars.append(w)
            grads.append(w_grad)
            vars.append(b)
            grads.append(b_grad)

        return vars,grads


class DFAConvnet(FAConvnet):

    def build_gradient(self):
        logit_grad = tf.gradients(self.batch_loss,self.y)[0]
        vars = []
        grads = []

        back_prob = logit_grad
        self._B = []

        # Fully-connected logit layer
        w,b = self._tf_layers[-1]
        y = self._tf_layers_output[-1]
        x = self._tf_layers_input[-1]
        a = self._tf_layers_preactivation[-1]
        d_z = back_prob
        w_grad = tf.matmul(tf.transpose(x), d_z)
        b_grad = tf.reduce_sum(d_z,axis=0)

        vars.append(w)
        grads.append(w_grad)
        vars.append(b)
        grads.append(b_grad)

        # Convolutional layers
        for i in reversed(range(len(self._tf_layers)-1)):
            w,b = self._tf_layers[i]
            y = self._tf_layers_output[i]
            x = self._tf_layers_input[i]
            a = self._tf_layers_preactivation[i]
            filters, kernel_size,stride = self.layer_dims[i]

            print("back_prop shape",str(back_prob.shape))
            print("y shape",str(y.shape))
            print("a shape",str(a.shape))
            print("x shape",str(x.shape))

            back_prob_shape = int(a.shape[1])*int(a.shape[2])*int(a.shape[3])
            B = self._backward_weight([self.output_dim,back_prob_shape],name="B_{}".format(i))
            self._B.append(B)
            dfa = tf.tensordot(back_prob,B,axes=[[1],[0]])
            dfa = tf.reshape(dfa,shape=tf.shape(a))
            print("dfa shape: ",str(dfa.shape))
            d_z = tf.multiply(dfa, self.grad_activation(a))

            w_grad = tf.nn.conv2d_backprop_filter(x,w.shape,out_backprop=d_z,strides=[1,stride,stride,1],padding="VALID")
            print("w_grad shape: ",str(w_grad.shape))

            b_grad = tf.reduce_sum(d_z,axis=[0,1,2])
            print("b_grad shape: ",str(b_grad.shape))

            vars.append(w)
            grads.append(w_grad)
            vars.append(b)
            grads.append(b_grad)

        return vars,grads

class mDFAConvnet(DFAConvnet):

    
    def build_preprocess(self,x):
        # x = tf.concat([x,-x],axis=-1)
        return x

    def build_constrain_op(self):
        op_list = []
        for i in range(1,len(self._tf_layers)):
            w,b = self._tf_layers[i]
            zeros = tf.zeros(shape=w.shape,dtype=tf.float32)+1e-6
            w_clipped = tf.maximum(w,zeros)
            clip_op = tf.assign(w,w_clipped)
            op_list.append(clip_op)
        return op_list

    def build_gradient(self):
        logit_grad = tf.gradients(self.batch_loss,self.y)[0]
        vars = []
        grads = []

        back_prob = logit_grad
        self._B = []

        # Fully-connected logit layer
        w,b = self._tf_layers[-1]
        y = self._tf_layers_output[-1]
        x = self._tf_layers_input[-1]
        a = self._tf_layers_preactivation[-1]
        d_z = back_prob
        w_grad = tf.matmul(tf.transpose(x), d_z)

        b_grad = tf.reduce_sum(d_z,axis=0)

        vars.append(w)
        grads.append(w_grad)
        vars.append(b)
        grads.append(b_grad)

        # Convolutional layers
        for i in reversed(range(len(self._tf_layers)-1)):
            w,b = self._tf_layers[i]
            y = self._tf_layers_output[i]
            x = self._tf_layers_input[i]
            a = self._tf_layers_preactivation[i]
            filters, kernel_size,stride = self.layer_dims[i]

            print("back_prop shape",str(back_prob.shape))
            print("y shape",str(y.shape))
            print("a shape",str(a.shape))
            print("x shape",str(x.shape))

            back_prob_shape = int(a.shape[1])*int(a.shape[2])*int(a.shape[3])
            B = self._backward_weight([self.output_dim,back_prob_shape],name="B_{}".format(i))
            self._B.append(B)
            dfa = tf.tensordot(back_prob,B,axes=[[1],[0]])
            dfa = tf.reshape(dfa,shape=tf.shape(a))
            print("dfa shape: ",str(dfa.shape))
            d_z = tf.multiply(dfa, self.grad_activation(a))
            print("d_z shape: ",str(d_z.shape))

            w_grad = tf.nn.conv2d_backprop_filter(x,w.shape,out_backprop=d_z,strides=[1,stride,stride,1],padding="VALID")
            print("w_grad shape: ",str(w_grad.shape))

            b_grad = tf.reduce_sum(d_z,axis=[0,1,2])
            print("b_grad shape: ",str(b_grad.shape))

            vars.append(w)
            grads.append(w_grad)
            vars.append(b)
            grads.append(b_grad)

        return vars,grads