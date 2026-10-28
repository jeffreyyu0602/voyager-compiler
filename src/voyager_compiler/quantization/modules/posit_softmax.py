import torch
from torch import nn


def _raise_missing_posit_tables():
    raise RuntimeError(
        "Posit softmax lookup tables are not included in this no-LFS branch. "
        "Run without --posit_exp, --posit_exp_shifted, or --posit_reciprocal."
    )


def _convert(input: torch.Tensor, values: torch.Tensor):
    # Keep 8 exponent bits and 14 fraction bits, which is the maximum number
    # of fraction bits for a 16-bit posit.
    if input.dtype == torch.bfloat16:
        indices = (input.view(torch.int16).int() << 7) & 0x3FFFFF
    else:
        raw_bits = input.float().view(torch.int32)
        indices = ((raw_bits >> 9) & 0x3FFFFF) | ((raw_bits & 0x1FF) != 0).int()
    return values[indices].to(input.dtype)


class PositSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, dim, posit_exp=None, posit_reciprocal=None):
        if posit_exp is None:
            exp_x = torch.exp(i)
        else:
            exp_x = _convert(i, posit_exp)
        exp_x_sum = torch.sum(exp_x, dim=dim, keepdim=True)

        if posit_reciprocal is None:
            output = exp_x / exp_x_sum
            ctx.save_for_backward(output, None, None)
        else:
            output = exp_x * _convert(exp_x_sum, posit_reciprocal)
            ctx.save_for_backward(output, exp_x, exp_x_sum)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, exp_x, exp_x_sum = ctx.saved_tensors

        if exp_x is None and exp_x_sum is None:
            grad_input = output * grad_output
            sum_grad = torch.sum(grad_input, dim=-1, keepdims=True)
            grad_input -= output * sum_grad
        else:
            grad_input = output * grad_output
            sum_grad = torch.sum(exp_x * grad_output, dim=-1, keepdims=True)
            deriv = torch.pow(2, torch.floor(torch.log2(exp_x_sum)) * -2 - 1)
            grad_input -= deriv * exp_x * sum_grad

        return grad_input, None, None, None

class Softmax(nn.Softmax):
    posit_exp: torch.Tensor
    posit_reciprocal: torch.Tensor

    def __init__(
        self,
        posit_exp=False,
        posit_exp_shifted=False,
        posit_reciprocal=False,
        dim=None,
        **kwargs,
    ):
        super().__init__(dim)
        self.posit_exp = None
        self.posit_reciprocal = None
        if posit_exp or posit_exp_shifted or posit_reciprocal:
            _raise_missing_posit_tables()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.amax(input, dim=self.dim, keepdim=True)
        return PositSoftmax.apply(
            input, self.dim, self.posit_exp, self.posit_reciprocal
        )
