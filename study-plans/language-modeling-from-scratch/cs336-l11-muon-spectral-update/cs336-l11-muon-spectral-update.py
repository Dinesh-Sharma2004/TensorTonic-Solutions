import torch

def muon_spectral_update(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    previous_momentum: torch.Tensor,
    momentum_coefficient: int | float,
    learning_rate: int | float,
) -> dict:
    """
    Returns a dict of tensors: new_parameter, new_momentum,
    orthogonalized_update.
    """

    # 1. Update momentum using the input dtype
    new_momentum = previous_momentum * momentum_coefficient + gradient

    # 2. Compute compact SVD in float32
    U, S, Vh = torch.linalg.svd(
        new_momentum.to(torch.float32),
        full_matrices=False
    )

    # 3. Drop singular values: O = U @ V^T
    orthogonalized_update = U @ Vh

    # FIX 1: Restore the update to the parameter's dtype
    orthogonalized_update = orthogonalized_update.to(parameter.dtype)

    # FIX 2: Update parameters using the correctly typed update
    new_parameter = parameter - learning_rate * orthogonalized_update

    return {
        "new_parameter": new_parameter,
        "new_momentum": new_momentum,
        "orthogonalized_update": orthogonalized_update
    }