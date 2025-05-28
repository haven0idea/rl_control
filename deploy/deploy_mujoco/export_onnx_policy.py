import torch
from legged_gym import LEGGED_GYM_ROOT_DIR

# load the trained policy jit model
policy_jit_path = f'{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/g1/policy_lstm_1.pt'
policy_jit_model = torch.jit.load(policy_jit_path)

#set the model to evalution mode
policy_jit_model.eval()
print(policy_jit_model.graph)

# creat a fake input to the model
test_input_tensor = torch.randn(1,45)  

#specify the path and name of the output onnx model
policy_onnx_model = f'{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/policy.onnx'

#export the onnx model
torch.onnx.export(policy_jit_model,               
                  test_input_tensor,       
                  policy_onnx_model,   # params below can be ignored
                  export_params=True,   
                  opset_version=11,     
                  do_constant_folding=True,  
                  input_names=['input'],    
                  output_names=['output'],  
                  )