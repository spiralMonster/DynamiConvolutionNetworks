import tensorflow as tf
from tensorflow.keras.layers import Layer

class InputSeparationLayer(Layer):
    
    def __init__(self,classes_config,**kwargs):
        super().__init__(**kwargs)
        self.classes_config=classes_config
        
    @tf.function 
    def call(self,predictions):
        pred=tf.argmax(predictions,axis=1)
        pred=tf.cast(pred,tf.int32)
        pred=predictions
        input_config={}
        
        for cls in self.classes_config.keys():
            ind=tf.where(pred==cls)
            
            if tf.size(ind)==0:
                input_config[cls]=tf.constant([],dtype=tf.int64)
                
            else:
                input_config[cls]=tf.reshape(ind,-1)
                
        return input_config
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                'classes_config':self.classes_config
            }
        )
        return config
                
                
        