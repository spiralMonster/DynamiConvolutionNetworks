import tensorflow as tf
from tensorflow.keras.layers import Layer
from hybrid_inception_skip_connection_layer import Hybrid_Inception_SkipConnection_Layer

class InputDrivenConvolutionDecider(Layer):
    def __init__(self,classes_config,max_output_cells,batch_size,**kwargs):
        super().__init__(**kwargs)
        self.classes_config=classes_config
        self.batch_size=batch_size
        self.max_output_cells=max_output_cells
        
        self.models= [Hybrid_Inception_SkipConnection_Layer(
                      num_output_layer_cells=i,
                      max_output_cells=self.max_output_cells,
                      batch_size=self.batch_size) 
                      for i in self.classes_config.values()]
        
    def call(self,inputs,input_config):
        outputs=[]
        
        for cls,inp_ind in input_config.items():
            if tf.size(inp_ind)!=0:
                inp=tf.gather(inputs,inp_ind)
                out=self.models[cls](inp)
                outputs.extend(out)
                
        final_out=tf.convert_to_tensor(outputs)
        return final_out
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                'classes_config':self.classes_config,
                'max_output_cells':self.max_output_cells,
                'batch_size':self.batch_size
            }
        )
        return config
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],self.max_output_cells)
        
                
        