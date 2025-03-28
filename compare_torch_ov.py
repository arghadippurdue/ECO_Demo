import torch
import openvino as ov
import numpy as np
import random
from pathlib import Path
from models.segformer import WeTr

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_ckpt(ckpt_path, ckpt_dict):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    for (k, v) in ckpt_dict.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k], strict=False)
    best_val = ckpt.get('best_val', 0)
    epoch_start = ckpt.get('epoch_start', 0)
    # print_log('Found checkpoint at {} with best_val {:.4f} at epoch {}'.
        # format(ckpt_path, best_val, epoch_start))
    return best_val, epoch_start

def create_segmenter(num_classes, device, backbone, ckpt_path):
    segmenter = WeTr(backbone, num_classes).to(device)
    # checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # segmenter.load_state_dict(checkpoint.get('segmenter', {}), strict=False)
    best_val, epoch_start = load_ckpt(ckpt_path, {'segmenter': segmenter})
    segmenter.eval()
    return segmenter


def compute_sad(tensor1, tensor2):
    print(tensor1.shape)
    print(tensor2.shape)
    return np.sum(np.abs(tensor1 - tensor2))

def main():
    set_seed()
    num_classes = 40
    device = torch.device('cpu')
    backbone = 'mit_b3'
    ckpt_path = "ckpt/b3_train/model-best.pth.tar"
    ov_model_path = Path("ov_model/mit_b3.xml")
    # ov_model_path = Path("ov_model/mit_b3_mmcv.xml")
    output_idx = 2
    
    # Load PyTorch model
    segmenter = create_segmenter(num_classes, device, backbone, ckpt_path)
    
    # Create dummy inputs
    input1 = torch.randn(1, 3, 468, 625, device=device)
    input2 = torch.randn(1, 3, 468, 625, device=device)
    inputs = [input1, input2]
    
    # Run inference on PyTorch model
    with torch.no_grad():
        torch_output = segmenter(inputs)[output_idx].cpu().numpy()
    
    # Load OpenVINO model
    core = ov.Core()
    compiled_model = core.compile_model(model=str(ov_model_path), device_name="CPU")
    # ov_input_name = compiled_model.inputs[0].get_any_name()
    # print(ov_input_name)
    
    # Prepare OpenVINO inputs
    # ov_inputs = {ov_input_name: np.stack([inp.numpy() for inp in inputs])}

    input_types = ['rgb', 'depth']
    ov_inputs = {}
    for idx, key in enumerate(input_types):
        ov_input_name = compiled_model.inputs[idx].get_any_name()
        # ov_input_name = compiled_model.inputs[(idx+1)%2].get_any_name()       # Swapping the inputs, SAD remains the same
        print(ov_input_name)
        ov_inputs[ov_input_name] = inputs[idx].float().cpu().numpy()
    
    # Run inference on OpenVINO model
    ov_results = compiled_model(ov_inputs)
    ov_output = ov_results[compiled_model.outputs[output_idx]]
    
    # Compute SAD
    sad = compute_sad(torch_output, ov_output)
    print(f"Sum of Absolute Differences (SAD) between PyTorch and OpenVINO outputs: {sad}")

if __name__ == "__main__":
    main()
