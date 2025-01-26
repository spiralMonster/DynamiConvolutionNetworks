import tensorflow as tf
from tensorflow.keras.layers import Layer

class CombinedOutputLayer(Layer):
    def __init__(self,batch_size,max_output_cells,**kwargs):
        super().__init__(**kwargs)
        self.batch_size=batch_size
        self.max_output_cells=max_output_cells
        
    def call(self,inputs,input_config):
        new_ind=tf.constant([],dtype=tf.int64)
        
        for ind in input_config.values():
            new_ind=tf.concat([new_ind,ind],axis=0)
            
        out=tf.zeros((self.batch_size,self.max_output_cells),dtype=tf.float32)
        
        for ind1,ind2 in enumerate(new_ind):
            out[ind2]=inputs[ind1]
            
        final_out=tf.cast(out,dtype=tf.float32)
        return final_out
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                'batch_size':self.batch_size,
                'max_output_cells':self.max_output_cells
            }
        )
        return config
        
    def compute_output_shape(self,input_shape):
        return input_shape
        
        