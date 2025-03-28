import torch
import openvino as ov
import numpy as np
import random
from pathlib import Path
from models.segformer import WeTr
import time

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
            state_dict = ckpt[k]
            # print(state_dict.keys())
            # Remove "module." prefix if model was trained with DataParallel
            new_state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
            # print(new_state_dict.keys())
            v.load_state_dict(new_state_dict, strict=True)
            # print(v.state_dict().keys())
            # v.load_state_dict(ckpt[k], strict=True)
    best_val = ckpt.get('best_val', 0)
    epoch_start = ckpt.get('epoch_start', 0)
    return best_val, epoch_start

def create_segmenter(num_classes, device, backbone, ckpt_path):
    segmenter = WeTr(backbone, num_classes).to(device)
    # segmenter = torch.nn.DataParallel(segmenter)
    best_val, epoch_start = load_ckpt(ckpt_path, {'segmenter': segmenter})
    segmenter.eval()
    return segmenter

def compute_sad(tensor1, tensor2):
    print(f"PyTorch Output Shape: {tensor1.shape}")
    print(f"OpenVINO Output Shape: {tensor2.shape}")
    return np.sum(np.abs(tensor1 - tensor2))

def convert_to_openvino(ov_model_path, segmenter, input1, input2):
    ov_model = ov.convert_model(
        segmenter,
        example_input=[input1, input2],
        input=[
            ("input1", ov.Shape([1, 3, 468, 625]), ov.Type.f32),
            ("input2", ov.Shape([1, 3, 468, 625]), ov.Type.f32)
        ]
    )
    # Save OpenVINO model
    ov.save_model(ov_model, ov_model_path, compress_to_fp16=True)
    print("OpenVINO model saved successfully!")
    return ov_model

def main():
    set_seed()
    num_classes = 40
    device = torch.device('cpu')
    backbone = 'mit_b3'
    ckpt_path = "ckpt/b3_train/model-best.pth.tar"
    ov_model_path = Path("ov_model/enc_dec_b3_torch_v1.xml")
    output_idx = 2
    use_async = False  # Set this flag to False for synchronous API mode
    ov_convert = True  # Set this flag to True to convert the model to OpenVINO

    # Load PyTorch model
    segmenter = create_segmenter(num_classes, device, backbone, ckpt_path)
    # for keys in segmenter.state_dict():
    #     print(keys)
    # # for name, param in segmenter.named_parameters():
    # #     print(name)
    # print(len(list(segmenter.state_dict().keys())))


    # Load tensors from the file
    file_path = "inputs.pth"
    loaded_data = torch.load(file_path)

    # Extract tensors
    input1 = loaded_data["input1"]
    input2 = loaded_data["input2"]

    # input1 = torch.randn(1, 3, 468, 625)
    # input2 = torch.randn(1, 3, 468, 625)
    # inputs = [input1, input2]
    
    # Run inference on PyTorch model
    with torch.no_grad():
        torch_output = segmenter(input1, input2)
        torch_output = torch_output[output_idx].cpu().numpy()

    if ov_convert:
        ## Convert to OpenVINO and save the model
        ov_model = convert_to_openvino(ov_model_path, segmenter, input1, input2)

        # Wait for 5 seconds to ensure the model is saved
        time.sleep(1)
    
    # Load OpenVINO model
    core = ov.Core()
    compiled_model = core.compile_model(model=str(ov_model_path), device_name="CPU")
    # print(compiled_model.outputs)
    # print(compiled_model.outputs[output_idx].index)

    # output_index = compiled_model.outputs[output_idx].index

    # Perform inference with OpenVINO (Asynchronous or Synchronous)
    if use_async:
        # Asynchronous Inference
        print("Running inference in Asynchronous mode")
        infer_request = compiled_model.create_infer_request()
        infer_request.infer(inputs={
            "input1": input1.numpy(),
            "input2": input2.numpy()
        })
        ov_output = infer_request.get_output_tensor(output_idx).data
    else:
        # Synchronous Inference
        print("Running inference in Synchronous mode")
        ov_output = compiled_model(inputs={
            "input1": input1.numpy(),
            "input2": input2.numpy()
        })[output_idx].data

    # Compute SAD (Sum of Absolute Differences)
    sad = compute_sad(torch_output, ov_output)
    print(f"Sum of Absolute Differences (SAD) between PyTorch and OpenVINO outputs: {sad}")

if __name__ == "__main__":
    main()
