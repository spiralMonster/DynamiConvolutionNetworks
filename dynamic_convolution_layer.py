import tensorflow as tf
from tensorflow.keras.layers import Layer
from basic_convolution_processor import  BasiConvolutionProcessor
from input_separation_layer import InputSeparationLayer
from input_driven_convolution_decider import InputDrivenConvolutionDecider
from combined_output_layer import CombinedOutputLayer

# classes_config: The number of subclasses of every class label ("class label":"number of subclasses")
# {
#     0:2,
#     1:5,
#     .....
# }

# pre_dcn_layers_config: The configurations of the layer before the DCN(Dynamic Convolution Network)
# {
#     "ConvolutionLayers":conv_layer_config-
#     "DenseLayers":dense_layer_config,                  
#     "LastLayer":last_layer_config
# }


class DynamiConvolutionLayer(Layer):
    def __init__(
        self,
        batch_size,
        classes_config,
        pre_dcn_layers_config,
        **kwargs
      
    ):
        super().__init__(**kwargs)
        self.batch_size=batch_size
        self.classes_config=classes_config
        self.max_output_cells=max(list(self.classes_config.values()))
        self.pre_dcn_layers_config=pre_dcn_layers_config
        
        # Configurations of Pre Dcn Layers:
        self.pre_dcn_conv_layer_config=self.pre_dcn_layers_config["ConvolutionLayers"]
        self.pre_dcn_dense_layer_config=self.pre_dcn_layers_config["DenseLayers"]
        self.pre_dcn_last_layer_config=self.pre_dcn_layers_config["LastLayer"]
        self.num_cells_last_layer_pre_dcn=len(self.classes_config)
        
        # PRE DCN LAYERS:
        self.pre_dcn_layer=BasiConvolutionProcessor(
            num_conv_layers=len(self.pre_dcn_conv_layer_config),
            conv_layer_config=self.pre_dcn_conv_layer_config,
            num_dense_layers=len(self.pre_dcn_dense_layer_config),
            dense_layer_config=self.pre_dcn_dense_layer_config,
            num_cells_last_layer=self.num_cells_last_layer_pre_dcn,
            last_layer_config=self.pre_dcn_last_layer_config
        )
        
        #DCN:
        
        #Input Separation for DCN:
        self.input_separation_layer=InputSeparationLayer(classes_config=self.classes_config)
        
        #Input Driven Convolution Decider Layer for DCN:
        self.input_driven_conv_layer_decider=InputDrivenConvolutionDecider(
            classes_config=self.classes_config,
            max_output_cells=self.max_output_cells
        )

        #Combined Output Layer for DCN:
        self.combined_output_layer=CombinedOutputLayer(
            batch_size=self.batch_size,
            max_output_cells=self.max_output_cells
        )
        
    def call(self,inputs):
        inp2,pred=self.pre_dcn_layer(inputs)
        input_config=self.input_separation_layer(pred)
        out=self.input_driven_conv_layer_decider(inp2,input_config)
        final_out=self.combined_output_layer(out,input_config)
        return pred,final_out


        