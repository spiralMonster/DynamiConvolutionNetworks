import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten

# conv_layer_config:
# [
#     {
#         'filters':,
#         'kernel_size':,
#         'padding':,
#         'activation':,
#         'kernel_initializer':
#     },
#
#
# ]

# dense_layer_config:
# [
#     {
#         'units':,
#         'activation':,
#         'kernel_initializer':
#     },
# ]


# last_layer_config:
# {
#     'activation':,
#     'kernel_initializer':
# }


# return:
#  inp2: Input for the DCN(Dynamic Convolutiuon Layer)
#  pred : First Predictions

class BasiConvolutionProcessor(Layer):
    def __init__(self,num_conv_layers,conv_layer_config,num_dense_layers,dense_layer_config,num_cells_last_layer,last_layer_config,**kwargs):
        super().__init__(**kwargs)
        self.num_conv_layers=num_conv_layers
        self.conv_layer_config=conv_layer_config
        self.num_dense_layers=num_dense_layers
        self.dense_layer_config=dense_layer_config
        self.num_cells_last_layer=num_cells_last_layer
        self.last_layer_config=last_layer_config
        self.conv_layers=[]
        self.dense_layers=[]
        
        #Convolution Layer Initializer:
        for ind in range(self.num_conv_layers):
            config=self.conv_layer_config[ind]
            
            self.conv_layers.append(
                Conv2D(
                    filters=config["filters"],
                    kernel_size=config["kernel_size"],
                    padding=config["padding"],
                    activation=config["activation"],
                    kernel_initializer=config["kernel_initializer"]
                )
            )

        
        #Dense Layer Initializer:
        for ind in range(self.num_dense_layers):
            config=self.dense_layer_config[ind]
            
            self.dense_layers.append(
                Dense(
                    units=config["units"],
                    activation=config["activation"],
                    kernel_initializer=config["kernel_initializer"]
                )
            )
            
        #Last Layer Initializer:   
        self.last_layer=Dense(
            units=self.num_cells_last_layer,
            activation=self.last_layer_config["activation"],
            kernel_initializer=self.last_layer_config["kernel_initializer"]
        )

    def call(self,x):
        
        for conv_layer in self.conv_layers:
            x=conv_layer(x)
            x=MaxPooling2D((2,2),padding="same")(x)
            
        inp2=x
        x=Flatten()(x)
        
        for dense_layer in self.dense_layers:
            x=dense_layer(x)
            
        pred=self.last_layer(x)
        
        return inp2,pred

    def get_config(self):
        config=super().get_config()
        config.update(
            {
                "num_conv_layers":self.num_conv_layers,
                "conv_layer_config":self.conv_layer_config,
                "num_dense_layers":self.num_dense_layers,
                "dense_layer_config":self.dense_layer_config,
                "num_cells_last_layer":self.num_cells_last_layer,
                "last_layer_config":self.last_layer_config  
                
            }
        )
        
        return config
            




        
        
            