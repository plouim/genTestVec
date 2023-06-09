from mmcv.cnn.bricks.conv_module import ConvModule
from torch.ao.quantization import fuse_modules
from mmcv.cnn import Swish
from mmdet.apis import init_detector
import torch
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
import numpy as np

# Set seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

def replace_act_func(module, name):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == Swish:
            print('replaced: ', name, attr_str)
            relu = nn.ReLU()
            setattr(module, attr_str, relu)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_act_func(immediate_child_module, name)

def get_current_name(model, layer):
    for n, m in model.named_modules():
        if m == layer:
            return n

def save_tensor2numpy(tensor, name, as_type, module=None):
    np_array = tensor.detach().numpy().astype(as_type)
    np_array.tofile(name+'.bin')
    print(f'SAVE INFO: SIZE: {tensor.shape},\t#OfElem: {torch.numel(tensor)}\tTOTAL: {np_array.size*np_array.itemsize} Bytes')
    print(f'SAVE INFO: Save {name}.bin')

# Define hook for fused layers
def pre_hook(module, input):
    # Save and Calcuate output_noq
    tmp_module = nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            bias=True
            )
    tmp_module.weight.data = module.weight.data 
    tmp_module.bias.data = module.bias.data
    output_noq = tmp_module(input[0])
    save_tensor2numpy(output_noq.permute(0,2,3,1), get_current_name(model, module)+'_output_f16_noq', 'float16')

    # 0. Preprocessing
    global scale_factor 
    global scale_weight
    global scale_bias

    i_max, i_min = input[0].max(), input[0].min()
    data_max = torch.max(i_max.abs(), i_min.abs())
    scale_factor = (2**7-1)/data_max

    w_max, w_min = module.weight.data.max(), module.weight.data.min()
    weight_max = torch.max(w_max.abs(), w_min.abs())
    scale_weight = (2**7-1)/weight_max

    # 1. Save the input(FP16), weight(FP16), bias(FP16)
    save_tensor2numpy(input[0].permute(0,2,3,1), get_current_name(model, module)+'_input_f16', 'float16')
    save_tensor2numpy(module.weight.data, get_current_name(model, module)+'_weight_f16', 'float16')
    save_tensor2numpy(module.bias.data, get_current_name(model, module)+'_bias_f16', 'float16')

    # 2. Quantize input, weight to INT8
    print(f"VALUE INFO: INPUT BEFORE: {input[0].max()}\t{input[0].min()}")
    module.weight.data = torch.round(module.weight.data * scale_weight, decimals=0)
    input = (torch.clip(torch.round(input[0] * scale_factor), min=-127, max=127),)
    print(f"VALUE INFO: INPUT AFTER: {input[0].max()}\t{input[0].min()}")

    # 3. Scaling bias
    module.bias.data = module.bias.data * (scale_factor* scale_weight)
    save_tensor2numpy(module.bias.data, get_current_name(model, module)+'_bias_f16_scale', 'float16')

    # 4. Save the quantized input(INT8), quantized weight(INT8)
    save_tensor2numpy(input[0].permute(0,2,3,1), get_current_name(model, module)+'_input_i8', 'int8')
    save_tensor2numpy(module.weight.data, get_current_name(model, module)+'_weight_i8', 'int8')

    # 5. Save scale parameters
    save_tensor2numpy(scale_factor, get_current_name(model, module)+'_scale_input_f16', 'float16')
    save_tensor2numpy(scale_weight, get_current_name(model, module)+'_scale_weight_f16', 'float16')
    save_tensor2numpy((scale_factor* scale_weight), get_current_name(model, module)+'_scale_iw_f16', 'float16')
    save_tensor2numpy(1/scale_factor, get_current_name(model, module)+'_scale_input_f16_inv', 'float16')
    save_tensor2numpy(1/scale_weight, get_current_name(model, module)+'_scale_weight_f16_inv', 'float16')
    save_tensor2numpy(1/(scale_factor* scale_weight), get_current_name(model, module)+'_scale_iw_f16_inv', 'float16')

    print(f"VALUE INFO: OUTPUT  NO Q: {output_noq.max()}\t{output_noq.min()}")
    print()

    return input

# Define hook for fused layers
def hook(module, input, output):
    # 0. Preprocessing
    global scale_factor
    global scale_weight
    global scale_bias
    i_max, i_min = output.max(), output.min()
    data_max = torch.max(i_max.abs(), i_min.abs())
    scale_output = (2**7-1)/data_max

#      # Rescaling bias
    #  module.bias.data = module.bias.data / scale_bias

    # 1. Quantize the output to INT8
    print(f"VALUE INFO: OUTPUT BEFORE: {output.max()}\t{output.min()}")
    output = output * scale_output
    output = torch.round(output, decimals=0)
    print(f"VALUE INFO: OUTPUT QUANT : {output.max()}\t{output.min()}")

    # 2. Save the output(INT8)
    save_tensor2numpy(output.permute(0,2,3,1), get_current_name(model, module)+'_output_i8', 'int8')

    # 3. Dequantize the output to FP16
    output = output / scale_output / scale_weight / scale_factor
    print(f"VALUE INFO: OUTPUT AFTER: {output.max()}\t{output.min()}")

    # 4. Save the output(FP16)
    save_tensor2numpy(output.permute(0,2,3,1), get_current_name(model, module)+'_output_f16', 'float16')

    # 5. Save the scale parameter
    save_tensor2numpy(scale_output, get_current_name(model, module)+'_scale_output_f16', 'float16')
    save_tensor2numpy(1/scale_output, get_current_name(model, module)+'_scale_output_f16_inv', 'float16')


    print('===============================================================================')

    return output

# Define hook for final output layer
def hook_for_final_layer(module, input, output):
    # 1. Save input, weight, output
    save_tensor2numpy(input[0].permute(0,2,3,1), get_current_name(model, module)+'_input_f16', 'float16')
    save_tensor2numpy(output.permute(0,2,3,1), get_current_name(model, module)+'_output_noq_f16', 'float16')
    save_tensor2numpy(module.weight.data, get_current_name(model, module)+'_weight_f16', 'float16')
    save_tensor2numpy(module.bias.data, get_current_name(model, module)+'_bias_f16', 'float16')
    
scale_weight = 0
scale_factor = 0
scale_bias = 0

cfg = './model/yolox_nano/yolox_v8_nano_8xb8-300e_dac.py'
ckpt = './model/yolox_nano/epoch_300.pth'

input_size = (32, 32)
model = init_detector(cfg, ckpt, device='cpu')
input = torch.randn((1, 3, *input_size)).float()

# Replace SiLU to ReLU
replace_act_func(model, 'model')

# Fusing conv, bn, relu
model.eval()
for m in model.modules():
    if isinstance(m, ConvModule):
        fuse_modules(m, [['conv', 'bn', 'activate']], inplace=True)
# Attach the hook to each module in the model
        # ConvModule structure after fusing conv, bn, relu layers
        #    (0): ConvModule(
        #      (conv): ConvReLU2d(
        #        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #        (1): ReLU()
        #      )
        #      (bn): Identity()
        #      (activate): Identity()
        #   )
for module in model.modules():
    if isinstance(module, ConvModule):
        module.conv[0].register_forward_pre_hook(pre_hook)
        module.conv[0].register_forward_hook(hook)

# Attach the hook to final module in the model
for name, module in model.named_modules():
    if name == 'bbox_head.multi_level_conv_cls':
        module[0].register_forward_hook(hook_for_final_layer)
        module[1].register_forward_hook(hook_for_final_layer)
        module[2].register_forward_hook(hook_for_final_layer)
    elif name == 'bbox_head.multi_level_conv_reg':
        module[0].register_forward_hook(hook_for_final_layer)
        module[1].register_forward_hook(hook_for_final_layer)
        module[2].register_forward_hook(hook_for_final_layer)
    elif name == 'bbox_head.multi_level_conv_obj':
        module[0].register_forward_hook(hook_for_final_layer)
        module[1].register_forward_hook(hook_for_final_layer)
        module[2].register_forward_hook(hook_for_final_layer)

# Run the model with some dummy input
output = model(input)
#  print(model)
