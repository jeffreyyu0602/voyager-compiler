import collections.abc
from itertools import repeat
from typing import Any, Optional

import torch
import torch.nn as nn

__all__ = [
    "get_arg_value",
    "get_conv_bn_layers",
]


def get_arg_value(
    node: torch.fx.Node,
    arg_number: int,
    kwarg_name: Optional[str] = None,
    default=None,
) -> Any:
    return (
        node.args[arg_number]
        if len(node.args) > arg_number
        else node.kwargs.get(kwarg_name, default)  # type: ignore[arg-type]
    )


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_pair = _ntuple(2, "_pair")


def get_conv_bn_layers(model):
    layers = []
    module_names = list(model._modules)
    for k, name in enumerate(module_names):
        if len(list(model._modules[name]._modules)) > 0:
            conv_bn_pairs = get_conv_bn_layers(model._modules[name])
            layers.extend([
                [f'{name}.{conv}', f'{name}.{bn}'] for conv, bn in conv_bn_pairs
            ])
        elif (
            isinstance(model._modules[name], nn.BatchNorm2d)
            and isinstance(model._modules[module_names[k-1]], nn.Conv2d)
        ):
            layers.append([module_names[k-1], name])
    return layers
