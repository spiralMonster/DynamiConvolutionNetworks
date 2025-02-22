import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv2D,LayerNormalization,Add

class SkipConnectionLayer(Layer):
    def __init__(self,filters,kernel_size,**kwargs):
        super().__init__(**kwargs)
        
        self.filters=filters
        self.kernel_size=kernel_size

        self.level0=Conv2D(self.filters,self.kernel_size,activation='relu',padding='same',kernel_initializer='he_uniform')
        self.level1=Conv2D(self.filters,self.kernel_size,activation='relu',padding='same',kernel_initializer='he_uniform')
        self.level2=Conv2D(self.filters,self.kernel_size,activation='relu',padding='same',kernel_initializer='he_uniform')
        self.level3=Conv2D(self.filters,self.kernel_size,activation='relu',padding='same',kernel_initializer='he_uniform')
        
        self.layer_normalization_level1=LayerNormalization()
        self.layer_normalization_level2=LayerNormalization()
        
    def call(self,inputs):
        inputs=self.level0(inputs)
        out1=self.level1(inputs)
        inp2=Add()([out1,inputs])
        inp2=self.layer_normalization_level1(inp2)
        
        out2=self.level2(inp2)
        inp3=Add()([out2,out1,inputs])
        inp3=self.layer_normalization_level2(inp3)
        
        out3=self.level3(inp3)
        return out3
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],self.filters)
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                'filters':self.filters,
                'kernel_size':self.kernel_size
            }
        )
        return config
        