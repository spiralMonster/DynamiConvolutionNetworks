import tensorflow as tf
from tensorflow.keras.layers import Layer

class InputSeparationLayer(Layer):
    
    def __init__(self,classes_config,**kwargs):
        super().__init__(**kwargs)
        self.classes_config=classes_config
        
    def collect_indices_by_class(self,ind):
        return tf.reshape(ind,[-1])

    def handle_absence_of_class_indices(self):
        return tf.constant([],tf.int64)
    
    
    def call(self,predictions):
        pred=tf.argmax(predictions,axis=1)
        pred=tf.cast(pred,tf.int32)
        input_config={}
        
        for cls in self.classes_config.keys():
            ind=tf.where(pred==cls)
            
            input_config[cls]=tf.cond(
                pred=tf.size(ind)>0,
                true_fn=lambda : self.collect_indices_by_class(ind),
                false_fn=self.handle_absence_of_class_indices
            )

                
        return input_config
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                'classes_config':self.classes_config
            }
        )
        return config
                
                
        