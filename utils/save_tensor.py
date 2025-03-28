import torch

# Create two float32 tensors
tensor1 = torch.rand(1, 40, 117, 157, dtype=torch.float32)  # Example tensor 1
tensor2 = torch.rand(1, 40, 117, 157, dtype=torch.float32)  # Example tensor 2

# Save tensors to a file
file_path = "tensors.pth"
torch.save({"tensor1": tensor1, "tensor2": tensor2}, file_path)
print(f"Tensors saved to {file_path}")

# Load tensors from the file
loaded_data = torch.load(file_path)

# Extract tensors
loaded_tensor1 = loaded_data["tensor1"]
loaded_tensor2 = loaded_data["tensor2"]

# Verify if loaded tensors match the original ones
is_same_tensor1 = torch.allclose(tensor1, loaded_tensor1, atol=1e-6)
is_same_tensor2 = torch.allclose(tensor2, loaded_tensor2, atol=1e-6)

# Print loaded tensors
print("Loaded Tensor 1:\n", loaded_tensor1)
print("Loaded Tensor 2:\n", loaded_tensor2)

# Check equality
if is_same_tensor1 and is_same_tensor2:
    print("✅ Tensors are successfully saved and loaded without modification!")
else:
    print("❌ Mismatch detected between saved and loaded tensors!")
