import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv2D,Dense,MaxPooling2D,LayerNormalization,Add

class InceptionLayer(Layer):
    
    def __init__(self,filters,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.level1A=Conv2D(self.filters//4,(7,7),activation='relu',padding='same',kernel_initializer='he_uniform')
        self.level1B=Conv2D(self.filters//4,(5,5),activation='relu',padding='same',kernel_initializer='he_uniform')
        
        self.level2A=Conv2D(self.filters//2,(5,5),activation='relu',padding='same',kernel_initializer='he_uniform')
        self.level2B=Conv2D(self.filters//2,(3,3),activation='relu',padding='same',kernel_initializer='he_uniform')
        
        self.level3=Conv2D(self.filters,(3,3),activation='relu',padding='same',kernel_initializer='he_uniform')
        
        self.layer_normalization_level1=LayerNormalization()
        self.layer_normalization_level2=LayerNormalization()
        
    def call(self,inputs):
        
        x=self.level1A(inputs)
        y=self.level1B(inputs)
        z=Add()([x,y])
        z=self.layer_normalization_level1(z)
        
        x=self.level2A(z)
        y=self.level2B(z)
        z=Add()([x,y])
        z=self.layer_normalization_level2(z)
        
        z=self.level3(z)
        
        return z
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],self.filters)
        
    def get_config(self):
        
        config=super().get_config()
        config.update({
           'filters':self.filters
        })
        return config
        
        
        
        



    
