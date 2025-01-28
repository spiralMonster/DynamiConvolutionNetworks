import tensorflow as tf
from tensorflow.keras.layers import Layer
from hybrid_inception_skip_connection_layer import Hybrid_Inception_SkipConnection_Layer

class InputDrivenConvolutionDecider(Layer):
    def __init__(self,classes_config,max_output_cells,**kwargs):
        super().__init__(**kwargs)
        self.classes_config=classes_config
        self.max_output_cells=max_output_cells
        
        self.models= [Hybrid_Inception_SkipConnection_Layer(
                      num_output_layer_cells=i,
                      max_output_cells=self.max_output_cells) 
                      for i in self.classes_config.values()]
    
    def handle_present_indices(self,cls,inputs,inp_ind):
        inp=tf.gather(inputs,inp_ind)
        batch_size=inp.shape[0]
        model=self.models[cls]
        out=model(inp)
        out=tf.convert_to_tensor(out)
        out=tf.expand_dims(out,axis=1)
        return out
        
    def handle_absent_indices(self,cls,inputs,inp_ind):
        out=[]
        return out
    
    def call(self,inputs,input_config):
        outputs=[]
        
        for cls,inp_ind in input_config.items():
            out=tf.cond(
                tf.greater(tf.size(inp_ind),0),
                true_fn=lambda :self.handle_present_indices(cls,inputs,inp_ind),
                false_fn=lambda :self.handle_absent_indices(cls,inputs,inp_ind)
            )
            
            outputs.extend(out)
                
        final_out=outputs[0]
        
        for tensor in outputs[1:]:
            final_out=tf.concat([final_out,tensor],axis=0)
            
        return final_out
        
    def get_config(self):
        config=super().get_config()
        config.update(
            {
                'classes_config':self.classes_config,
                'max_output_cells':self.max_output_cells
            }
        )
        return config
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],self.max_output_cells)
        
                
        