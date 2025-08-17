import torch
import torch.nn as nn
from collections import OrderedDict


def create_model_from_config(
    architecture: list, input_shape: tuple = (3, 32, 32)
) -> nn.Sequential:
    """
    Builds a PyTorch model dynamically from a configuration list.

    Args:
        architecture: A list of dictionaries, where each dict describes a layer.
        input_shape: The shape of a single input image (C, H, W).

    Returns:
        A torch.nn.Sequential model.
    """
    layers = OrderedDict()
    in_features = None  # We'll calculate this dynamically

    # First, build the convolutional part to determine the flattened size
    conv_layers = []
    for i, layer_config in enumerate(architecture):
        layer_type = layer_config["type"]
        if layer_type in ["Linear", "Dropout"]:
            # Stop when we hit the first dense layer
            break

        params = layer_config.get("params", {})
        # Find the layer class in torch.nn and instantiate it
        layer_class = getattr(nn, layer_type)
        conv_layers.append(layer_class(**params))

    # Use a dummy tensor to find the output size of the conv part
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        conv_part = nn.Sequential(*conv_layers)
        conv_output = conv_part(dummy_input)
        in_features = conv_output.flatten().shape[0]

    # Now, build the full model layer by layer
    for i, layer_config in enumerate(architecture):
        layer_type = layer_config["type"]
        params = layer_config.get("params", {})
        layer_name = f"{layer_type.lower()}_{i}"

        # If it's the first Linear layer, set its in_features
        if layer_type == "Linear" and "in_features" not in params:
            if in_features is None:
                raise ValueError("Could not determine in_features for Linear layer.")
            params["in_features"] = in_features

        layer_class = getattr(nn, layer_type)
        layer = layer_class(**params)
        layers[layer_name] = layer

        # The output of this linear layer is the input to the next
        if layer_type == "Linear":
            in_features = params["out_features"]

    return nn.Sequential(layers)
