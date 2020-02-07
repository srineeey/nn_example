"""Neural Network class"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

"""
the neural network class inherits from nn.Module
the default functions
init (constructor) and forward
have to be defined manually
"""
class Net(nn.Module):
    """
    class constructor
    typically the neural networks variables and structure are initialized here
    super().__init__() is necessary to call the mother class constructor
    """
    def __init__( self, net_struct, input_size ):
        super(Net, self).__init__()
        self.net_struct = net_struct
        self.input_size = input_size
        #self.output_size = net_struct[-1]["layer_pars"]["out_features"]
        #self.input_size = net_struct[0]["layer_pars"]["in_features"]
        
        self.batch_size = 1
        

        
        #self.fc = []
        """
        Using regular list results in an empty parameter list
        nn.ModuleList() is a way to save layers in a python type list
        to avoid this error
        """
        self.fc = nn.ModuleList()

        """
        Construct neural network layers from net_struct dictionary
        """

        for layer in self.net_struct:
            print("Adding " + str(layer) + "\n")
            self.fc.append(layer["type"](**layer["layer_pars"]))
            
        self.init_weights(torch.nn.init.xavier_normal_)
        
        self.layer_sizes = self.calc_layer_sizes()
        
        self.output_size = self.layer_sizes[-1]


    """
    the forward function is called each the the network is propagating forward
    takes in input data and spits out the predicted output
    """
    def forward(self, input_data):

        x = input_data
        #print(x.shape)
        
        """
        iterate through all layers and perform calculation
        """
        for layer_i in range(len(self.fc)):
            #print(layer_i)
            #print(x.shape)
            #print(self.net_struct[layer_i]["type"])
            #if self.net_struct[layer_i]["type"] == nn.Linear:
            if self.net_struct[layer_i]["type"] == nn.Linear or self.net_struct[layer_i]["type"] == nn.BatchNorm1d:
                #print(np.prod(self.layer_sizes[layer_i]))
                x = x.reshape( (-1,np.prod(self.layer_sizes[layer_i])) )
            #print(x.shape)
            z = self.fc[layer_i](x)
            if "act_func" in self.net_struct[layer_i]:
                x = self.net_struct[layer_i]["act_func"](z)
            else:
                x = z

            #print(x)
            

        return x
    """
    initialize weights
    Using initialization routine from the torch.nn.init package
    """
    def init_weights(self, init_routine):
        for i, layer in enumerate(self.fc):
            if type(layer) == nn.Linear:
                #torch.nn.init.xavier_normal_(layer.weight)
                init_routine(layer.weight)
                #if self.net_struct[i]["bias"] == True:
                    #layer.bias.data.fill_(0.01)

    """
    Basic method(s) to quickly display network structure, input, output
    """
    
    
    
    def calc_layer_sizes(self, input_shape = None, net_struct = None):
        if input_shape == None:
            input_shape = self.input_size
            
        if net_struct == None:
            net_struct = self.net_struct
        
        layer_sizes = [input_shape]
        for i in range(len(net_struct)):

            new_layer_size = []
            if net_struct[i]["type"] == nn.Linear:
                #print(net_struct[i]["type"])
                new_layer_size = net_struct[i]["layer_pars"]["out_features"]

            elif net_struct[i]["type"] == nn.Conv2d:
                kernel_shape = net_struct[i]["layer_pars"]["kernel_size"]
                stride = net_struct[i]["layer_pars"]["stride"]

                new_layer_size = []
                #print("layer " + str(i))
                for d in range(len(kernel_shape)):
                    prev_layer_l = int(layer_sizes[-1][d+1])
                    kernel_l = int(kernel_shape[d])
                    new_layer_size.append( (prev_layer_l - kernel_l)//stride + 1 )
                new_layer_size = [net_struct[i]["layer_pars"]["out_channels"]] + new_layer_size

            elif net_struct[i]["type"] == nn.MaxPool2d:
                kernel_shape = net_struct[i]["layer_pars"]["kernel_size"]
                stride = net_struct[i]["layer_pars"]["stride"]

                new_layer_size = []
                for d in range(len(kernel_shape)):
                    prev_layer_l = int(layer_sizes[-1][d+1])
                    kernel_l = int(kernel_shape[d])
                    new_layer_size.append(int( (prev_layer_l - kernel_l) + 1 )//stride)

                #new_layer_size = [(layer_sizes[-1][d+1] - kernel_shape[d])/stride + 1 for d in range(len(kernel_shape))]

                prev_channels = layer_sizes[-1][0]
                new_layer_size = [prev_channels] + new_layer_size
                
            elif net_struct[i]["type"] == nn.BatchNorm1d or net_struct[i]["type"] == nn.Dropout or net_struct[i]["type"] == nn.Softmax:
                new_layer_size = layer_sizes[-1]

            layer_sizes.append(new_layer_size)

        return layer_sizes
    
    def show_net_struct(self):
        print("Network Architecture:\n")
        for layer in self.net_struct:
            print( "Layer: {}".format(layer))

    def show_layers(self):
        for layer in self.fc:
            print(layer)

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size

    def get_net_struct(self):
        return self.net_struct
    

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
