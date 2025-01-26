import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,Flatten,BatchNormalization,Input,MaxPooling2D,Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from inception_layer import InceptionLayer
from skip_connection_layer import SkipConnectionLayer

class Hybrid_Inception_SkipConnection_Layer(Layer):
    
    def __init__(self,num_output_layer_cells,max_output_cells,batch_size,**kwargs):
        super().__init__(**kwargs)
        self.num_output_layer_cells=num_output_layer_cells
        self.max_output_cells=max_output_cells
        self.batch_size=batch_size
        self.padding_length=self.max_output_cells-self.num_output_layer_cells
        
        #Inception Layers:
        self.incept1=InceptionLayer(filters=16)
        self.incept2=InceptionLayer(filters=32)
        self.incept3=InceptionLayer(filters=64)
        
        #Skip Connection Layers:
        self.skip1=SkipConnectionLayer(filters=16,kernel_size=(3,3))
        self.skip2=SkipConnectionLayer(filters=32,kernel_size=(5,5))
        self.skip3=SkipConnectionLayer(filters=64,kernel_size=(3,3))
        
        #Dense Layers:
        self.dense1=Dense(units=1024,activation='relu',kernel_initializer='he_uniform')
        self.dense2=Dense(units=256,activation='relu',kernel_initializer='he_uniform')
        self.dense3=Dense(units=4*self.num_output_layer_cells,activation='relu',kernel_initializer='he_uniform')
        self.dense4=Dense(units=2*self.num_output_layer_cells,activation='relu',kernel_initializer='he_uniform')
        self.dense5=Dense(units=self.num_output_layer_cells,activation='softmax',kernel_initializer='glorot_uniform')
        
        #Batch Normalization Layers:
        self.batch1=BatchNormalization()
        self.batch2=BatchNormalization()
        self.batch3=BatchNormalization()
        
    def call(self,input):
               
        x=self.incept1(input)
        x=MaxPooling2D((2,2),padding='same')(x)
        x=self.skip1(x)
        x=MaxPooling2D((2,2),padding='same')(x)
        x=self.batch1(x)
        
        x=self.incept2(x)
        x=MaxPooling2D((2,2),padding='same')(x)
        x=self.skip2(x)
        x=MaxPooling2D((2,2),padding='same')(x)
        x=self.batch2(x)
        
        x=self.incept3(x)
        x=self.skip3(x)
        x=MaxPooling2D((2,2),padding='same')(x)
        x=self.batch3(x)
        
        
        x=Flatten()(x)
        x=self.dense1(x)
        x=self.dense2(x)
        x=Dropout(0.25)(x)
        x=self.dense3(x)
        x=Dropout(0.2)(x)
        x=self.dense4(x)
        x=self.dense5(x)

        if self.padding_length>0:
            padding=tf.zeros((self.batch_size,self.padding_length),dtype=tf.float32)
            x=tf.concat([x,padding])
        
        return x
        
    def get_config(self):
        config=super().get_config()
        config.update({
            'num_output_layer_cells':self.num_output_layer_cells,
            'max_output_cells':self.max_output_cells,
            'batch_size':self.batch_size
        })
        return config
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],self.max_output_cells)
        
        
        