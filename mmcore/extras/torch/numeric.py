import torch

import torch

from mmcore.numeric.fdm import _MACHINE_DECIMALS_SQ,_MACHINE_EPS_SQ
def grad_scalar_valued(fun, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Computes the numerical gradient of the function `fun` at point `x`
    using the central finite difference method.

    :param fun: A function that takes a tensor x of shape (..., N)
                and returns a tensor of shape (..., ) or (..., 1)
    :param x: Input tensor of shape (..., N)
    :param eps: A small float number for finite differences (default: 1e-6)
    :return: Tensor of gradients with shape (..., N)
    """
    # Create a tensor to hold the gradient with the same shape as x.

    grad = torch.zeros_like(x)

    # Loop over the last dimension (the N features)
    for i in range(x.shape[-1]):
        # Clone x to avoid in-place modifications
        x_plus = x.clone()
        x_minus = x.clone()

        # Perturb only the i-th coordinate
        x_plus[..., i] += eps
        x_minus[..., i] -= eps

        # Evaluate the function at the perturbed points.
        # fun(x_plus) and fun(x_minus) have shape (..., 1)
        f_plus = fun(x_plus)
        f_minus = fun(x_minus)

        # Compute the central finite difference.
        # Squeeze the last dimension to match the shape of grad[..., i] (which is (...,))
        grad[..., i] = ((f_plus - f_minus) / (2 * eps)).squeeze(-1)

    return grad


def fdm_decorator(epsilon=1e-5):
    r"""
    A decorator to wrap a black‐box function (which accepts and returns a torch.Tensor)
    so that it is differentiable via automatic numerical differentiation using a finite‐difference
    method in the backward pass.

    The wrapped function f: Tensor -> Tensor must have the property that the output shape
    is strictly determined by the input shape.

    Args:
        epsilon (float): A small positive number used as the finite difference step.

    Returns:
        A decorator which converts the black‐box function into one that supports autograd.

    Raises:
        ValueError: If epsilon is not positive.
        TypeError: If the decorated object is not callable.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    def decorator(func):
        if not callable(func):
            raise TypeError("The decorated object must be callable")

        class FiniteDiffFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                # Ensure the input is a torch.Tensor.
                if not isinstance(input, torch.Tensor):
                    raise TypeError("Input must be a torch.Tensor")
                # Save the input for the backward pass.
                ctx.save_for_backward(input)
                # Save the black-box function and finite difference step in the context.
                ctx.func = func
                ctx.epsilon = epsilon
                # Compute the output using the provided function.
                output = func(input)
                if not isinstance(output, torch.Tensor):
                    raise TypeError("The black-box function must return a torch.Tensor")
                return output

            @staticmethod
            def backward(ctx, grad_output):
                """
                Compute the gradient with respect to the input by approximating the Jacobian
                using central finite differences.

                For f: ℝⁿ → ℝᵐ, we approximate the (i,j) entry of the Jacobian J as:

                    J[j, i] ≈ [f(x + e_i * ε)[j] - f(x - e_i * ε)[j]] / (2ε)

                Then, using the chain rule, the gradient with respect to x is:

                    grad_input = Jᵀ * grad_output_flat

                where grad_output_flat is grad_output flattened to a 1D tensor.
                """
                # Retrieve the saved input tensor.
                (input,) = ctx.saved_tensors
                func = ctx.func
                epsilon = ctx.epsilon

                if not isinstance(grad_output, torch.Tensor):
                    raise TypeError("grad_output must be a torch.Tensor")

                # Compute the base output to determine output size.
                with torch.no_grad():
                    base_output = func(input)
                # Flatten the input and base output.
                input_flat = input.view(-1)
                base_output_flat = base_output.view(-1)
                num_input = input_flat.numel()
                num_output = base_output_flat.numel()

                # Initialize a list to store the derivative (a column of the Jacobian) for each input element.
                jacobian_cols = []

                # Loop over each element of the flattened input.
                for i in range(num_input):
                    # Create a perturbation tensor of zeros with the same shape as input.
                    perturb = torch.zeros_like(input)
                    # Add epsilon to the i-th element (using the flattened view).
                    perturb.view(-1)[i] = epsilon

                    # Compute f(x + perturb) and f(x - perturb) without tracking gradients.
                    with torch.no_grad():
                        output_plus = func(input + perturb)
                        output_minus = func(input - perturb)
                    # Flatten the outputs.
                    output_plus_flat = output_plus.view(-1)
                    output_minus_flat = output_minus.view(-1)
                    # Use the central difference formula.
                    derivative = (output_plus_flat - output_minus_flat) / (
                            2.0 * epsilon
                    )
                    # Append this derivative column (of shape (num_output,)) to our list.
                    jacobian_cols.append(derivative)

                # Stack the columns to form the Jacobian matrix.
                # The resulting tensor has shape (num_output, num_input) where
                # each column corresponds to the gradient of the output with respect to one input element.
                jacobian = torch.stack(
                    jacobian_cols, dim=1
                )  # Shape: (num_output, num_input)

                # Flatten grad_output to shape (num_output,).
                grad_output_flat = grad_output.view(-1)
                # Apply the chain rule: grad_input = Jᵀ * grad_output.
                grad_input_flat = torch.matmul(
                    jacobian.t(), grad_output_flat
                )  # Shape: (num_input,)
                # Reshape grad_input to the original input shape.
                grad_input = grad_input_flat.view_as(input)
                return grad_input

        def wrapped_function(input):
            return FiniteDiffFunction.apply(input)

        return wrapped_function

    return decorator


# === Example Usage ===
if __name__ == "__main__":
    # Suppose we have a complex black-box function (whose internal operations are not autograd‐tracked)
    # that takes and returns a torch.Tensor. For illustration, we define one that computes:
    #   f(x) = sin(x) + x^2
    # but assume that this function is implemented externally.

    # Наша грязная реализация
    import numpy as np


    @fdm_decorator(epsilon=1e-5)
    def black_box_function(x: torch.Tensor) -> torch.Tensor:
        # The function can perform arbitrarily complex operations.
        xx = x.detach().numpy()
        return torch.from_numpy(np.sin(xx) + xx ** 2)


    # Прекрасная реализация с помощью функций торча
    def not_black_box_function(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + x ** 2


    # Black-Box implementation
    # ------------------------------------------------------------------------------------------------------------------
    # Create an example input tensor.
    x1 = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
    # Compute the output.
    y1 = black_box_function(x1)
    # Compute a scalar loss (e.g. the sum of outputs).
    loss = y1.sum()
    # Backpropagate.
    loss.backward()
    print("Input:\n", x1)
    print("Output:\n", y1)
    print("Gradient computed via custom finite differences:\n", x1.grad)

    # Pure torch implementation:
    # ------------------------------------------------------------------------------------------------------------------
    # Create an example input tensor.
    x2 = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
    # Compute the output.
    y2 = not_black_box_function(x2)
    # Compute a scalar loss (e.g. the sum of outputs).
    loss = y2.sum()
    # Backpropagate.
    loss.backward()
    # Print the results.
    print("Input:\n", x2)
    print("Output:\n", y2)
    print("Gradient computed via autograd:\n", x2.grad)

    print("\n\nThat's what I needed to prove:", "\n(black box) ", x1.grad, "\n(pure torch)", x2.grad,
          "\n\n>>> torch.all(x1.grad-x2.grad,0) # final check\n", torch.all(x1.grad - x2.grad, 0))

