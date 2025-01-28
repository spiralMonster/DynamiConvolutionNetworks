import tensorflow as tf
from tensorflow.keras.models import Model
from dynamic_convolution_layer import DynamiConvolutionLayer

class DynamiConvolutionModel(Model):
    def __init__(self,classes_config,pre_dcn_layers_config,batch_size,**kwargs):
        super().__init__(**kwargs)
        self.classes_config=classes_config
        self.pre_dcn_layers_config=pre_dcn_layers_config
        self.batch_size=batch_size
        
        #Dynamic Convolution Layer:
        self.dynamic_convolution_layer=DynamiConvolutionLayer(
            classes_config=self.classes_config,
            pre_dcn_layers_config=self.pre_dcn_layers_config,
            batch_size=self.batch_size
        )
        
    @tf.function
    def forward_pass(self,inputs):
        return self.dynamic_convolution_layer(inputs)
        
    def call(self,inputs,training=False):
        return self.forward_pass(inputs)
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                "classes_config":self.classes_config,
                "pre_dcn_layers_config":self.pre_dcn_layers_config,
                "batch_size":self.batch_size
            }
        )
        return config
        

        