from mmcv.cnn import Swish
from mmdet.apis import init_detector
import torch
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

scale_weight = 0
scale_factor = 0
cfg = './model/yolox_s/yolox_s_8xb8-300e_dac.py'
ckpt = './model/yolox_s/epoch_300.pth'

input_size = (32, 32)
model = init_detector(cfg, ckpt, device='cpu')
input = torch.randn((1, 3, *input_size)).float()


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

# Define a pre hook 
def pre_hook(module, input):
    #################################################################################################
    #  1. Save the output for output_layer
    tmp_module = nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            bias=False
            )
    tmp_module.weight.data = module.weight.data
    tmp = tmp_module(input[0])
    save_tensor2numpy(tmp.permute(0,2,3,1), get_current_name(model, module)+'_output_f16_noq', 'float16')
    #################################################################################################
    #  2.1. Save the weight(FP16)
    save_tensor2numpy(module.weight.data, get_current_name(model, module)+'_weight_f16', 'float16')

    #  2.2. Quantize the weight to INT8
    print(f"VALUE INFO: INPUT BEFORE: {input[0].max()}\t{input[0].min()}")
    global scale_weight
    w_max, w_min = module.weight.data.max(), module.weight.data.min()
    weight_max = torch.max(w_max.abs(), w_min.abs())
    scale_weight = (2**7-1)/weight_max
    module.weight.data = module.weight.data * scale_weight
    module.weight.data = torch.round(module.weight.data, decimals=0)

    # 2.3. Save the quantized weight(INT8)
    save_tensor2numpy(module.weight.data, get_current_name(model, module)+'_weight_i8', 'int8')

    #################################################################################################
   
    save_tensor2numpy(input[0].permute(0,2,3,1), get_current_name(model, module)+'_input_f16', 'float16')

    # 3.2. Quantize the input to INT8
    global scale_factor
    i_max, i_min = input[0].max(), input[0].min()
    data_max = torch.max(i_max.abs(), i_min.abs())
    scale_factor = (2**7-1)/data_max
    input = (torch.clip(torch.round(input[0] * scale_factor), min=-127, max=127),)
    print(f"VALUE INFO: INPUT AFTER: {input[0].max()}\t{input[0].min()}")

    # 3.3. Save the quantized input(INT8)
    save_tensor2numpy(input[0].permute(0,2,3,1), get_current_name(model, module)+'_input_i8', 'int8')

    # 3.4. Save scale parameters
    save_tensor2numpy(scale_weight, get_current_name(model, module)+'_scale_weight_f16', 'float16')
    save_tensor2numpy(scale_factor, get_current_name(model, module)+'_scale_input_f16', 'float16')
    print()

    return input

# Define a hook 
def hook(module, input, output):
    global scale_factor
    global scale_weight
    i_max, i_min = output.max(), output.min()
    data_max = torch.max(i_max.abs(), i_min.abs())
    scale_output = (2**7-1)/data_max
    
    # 1. Quantize the output to INT8
    print(f"VALUE INFO: OUTPUT BEFORE: {output.max()}\t{output.min()}")
    output = output * scale_output
    output = torch.round(output, decimals=0)
    print(f"VALUE INFO: OUTPUT QUANT : {output.max()}\t{output.min()}")

    # 2. Save the output(INT8)
    save_tensor2numpy(output.permute(0,2,3,1), get_current_name(model, module)+'_output_i8', 'int8')

    # 3. Save the scale parameter
    save_tensor2numpy(scale_output, get_current_name(model, module)+'_scale_output_f16', 'float16')

    # 4. Dequantize the output to FP16
    output = output / scale_output / scale_weight / scale_factor
    print(f"VALUE INFO: OUTPUT AFTER: {output.max()}\t{output.min()}")

    # 5. Save the output(FP16)
    save_tensor2numpy(output.permute(0,2,3,1), get_current_name(model, module)+'_output_f16', 'float16')
    print(f"Conv Layer Bias: {module.bias}")
    print('===============================================================================')
    return output
    
def hook_for_BN(module, input, output):
    print(f"BN INPUT: {input[0].max()}\t{input[0].min()}")
    print(f"BN OUTPUT: {output.max()}\t{output.min()}")
    save_tensor2numpy(input[0].permute(0,2,3,1), get_current_name(model, module)+'_input_f16', 'float16')
    save_tensor2numpy(output.permute(0,2,3,1), get_current_name(model, module)+'_output_f16', 'float16')
    save_tensor2numpy(module.weight.data, get_current_name(model, module)+'_weight_f16', 'float16')
    save_tensor2numpy(module.bias.data, get_current_name(model, module)+'_bias_f16', 'float16')
def hook_for_ReLU(module, input, output):
    print(f"VALUE INFO: OUTPUT AFTER: {output.max()}\t{output.min()}")
    print('===============================================================================')

# Replace SiLU to ReLU
replace_act_func(model, 'model')

# Attach the hook to every module in the model
for name, module in model.named_modules():
    #  if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
    if isinstance(module, nn.Conv2d):
        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(hook)
    elif isinstance(module, nn.BatchNorm2d):
        module.register_forward_hook(hook_for_BN)
    #  elif isinstance(module, nn.ReLU):
        #  module.register_forward_hook(hook_for_ReLU)
    
model.eval()

# Run the model with some dummy input
output = model(input)
print(model)
