import torch.nn as nn
import torch

num_parallel = 2


class TokenExchange(nn.Module):
    def __init__(self):
        super(TokenExchange, self).__init__()

    def forward(self, x, mask, mask_threshold):
        # # x: [B, N, C], mask: [B, N, 1]
        # x0, x1 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        # x0[mask[0] >= mask_threshold] = x[0][mask[0] >= mask_threshold]
        # x0[mask[0] < mask_threshold] = x[1][mask[0] < mask_threshold]
        # x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
        # x1[mask[1] < mask_threshold] = x[0][mask[1] < mask_threshold]
        # return [x0, x1]

        ## AD: alternate implementation
        if not (isinstance(x, list) and isinstance(mask, list) and len(x) == 2 and len(mask) == 2):
            raise ValueError("x and mask should be lists of two tensors each.")

        x0, x1 = x  # Extract input tensors
        mask0, mask1 = mask  # Extract mask tensors

        # Ensure masks are boolean and properly expanded to match [B, N, C]
        mask0 = (mask0 >= mask_threshold).unsqueeze(-1).expand(-1, -1, x0.shape[-1])  # [B, N, C]
        mask1 = (mask1 >= mask_threshold).unsqueeze(-1).expand(-1, -1, x1.shape[-1])  # [B, N, C]

        # Convert mask to float for arithmetic operations
        mask0 = mask0.float()
        mask1 = mask1.float()

        # Apply exchange operation with corrected mask inversion
        x0_new = x0 * mask0 + x1 * (1.0 - mask0)
        x1_new = x1 * mask1 + x0 * (1.0 - mask1)

        return [x0_new, x1_new]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'ln_' + str(i), nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'ln_' + str(i))(x) for i, x in enumerate(x_parallel)]
