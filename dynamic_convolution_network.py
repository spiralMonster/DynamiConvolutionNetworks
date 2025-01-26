import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv2D,Dense,MaxPooling2D,Flatten,Input,BatchNormalization
from tensorflow.keras.models import Model
from input_separation import InputSeparationLayer
from input_driven_layer import InputDrivenConvolutionLayer
from combine_output import CombineOutputLayer

class DynamiConvolutionNetwork:
    
    def __init__(self,input_shape,classes_config,batch_size):
        self.input_shape=input_shape
        self.classes_config=classes_config
        self.batch_size=batch_size
        self.max_output_cells=max(list(self.classes_config.values()))
        
    def build(self):
        inp=Input(shape=self.input_shape,dtype=tf.float32)
        
        x=Conv2D(16,(5,5),activation='relu',padding='same',kernel_initializer='he_uniform')(inp)
        x=MaxPooling2D((2,2),padding='same')(x)
        x=BatchNormalization()(x)
         
        x=Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_uniform')(x)
        x=MaxPooling2D((2,2),padding='same')(x)
        x=BatchNormalization()(x)
         
        x=Conv2D(64,(5,5),activation='relu',padding='same',kernel_initializer='he_uniform')(x)
        x=MaxPooling2D((2,2),padding='same')(x)
        inp2=BatchNormalization()(x)

        x=Dense(256,activation='relu',kernel_initializer='he_uniform')(inp2)
        x=Dense(64,activation='relu',kernel_initializer='he_uniform')(x)
        pred1=Dense(len(list(self.classes_config.keys())),activation='softmax',kernel_initializer='glorot_uniform')(x)
        
        input_config=InputSeparationLayer(classes_config=self.classes_config)(pred1)
        
        out=InputDrivenConvolutionLayer(classes_config=self.classes_config,
                                        max_output_cells=self.max_output_cells,
                                        batch_size=self.batch_size)(inp2,input_config)
        
        final_out=CombineOutputLayer(input_config=input_config,
                                     batch_size=self.batch_size,
                                     max_output_cells=self.max_output_cells)(out)
        
        model=Model(inputs=inp,outputs=[pred1,final_out])
        return model
        

       
        
                
        


        
    
