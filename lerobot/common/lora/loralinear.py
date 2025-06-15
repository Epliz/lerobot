from lerobot.configs.lora import LoRaConfig

import torch
from torch import nn


class LoRaLinear(nn.Linear):
    """
    A linear layer with LoRa (Low-Rank Adaptation) support.
    This class extends the standard nn.Linear layer to include LoRa functionality.
    """

    def __init__(
        self,
        lora_config: LoRaConfig,
        in_features,
        out_features,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

        self.lora_config = lora_config
        self.alpha = lora_config.alpha

        self.lora_a = nn.Linear(
            in_features,
            lora_config.rank,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.lora_b = nn.Linear(
            lora_config.rank,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self._no_further_replacement = True  # Prevent further replacements in the model

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.zeros_(self.lora_b.weight)

    @staticmethod
    def replace(module: nn.Linear, lora_config: LoRaConfig, device="cuda"):
        """
        Replace a standard linear layer with a LoRaLinear layer.
        """
        dtype = module.weight.dtype
        has_bias = module.bias is not None
        requires_grad = module.weight.requires_grad

        lora_linear = LoRaLinear(
            lora_config=lora_config,
            in_features=module.in_features,
            out_features=module.out_features,
            bias=has_bias,
            device=device,
            dtype=dtype,
        )

        lora_linear.copy_module(module)

        lora_linear.weight.requires_grad = False
        lora_linear.lora_a.weight.requires_grad = requires_grad
        lora_linear.lora_b.weight.requires_grad = requires_grad

        if has_bias:
            lora_linear.bias.requires_grad = False
            lora_linear.lora_b.bias.requires_grad = requires_grad
        else:
            lora_linear.bias = None

        return lora_linear

    def copy_module(self, module: nn.Linear):
        """
        Copy the weights and bias from a standard linear layer to this LoRaLinear layer.
        """
        self.weight.data.copy_(module.weight.data)

        if self.bias is not None:
            self.bias.data.copy_(module.bias.data)
        else:
            self.bias = None

    def replace_back(self) -> nn.Linear:
        """
        Replace this LoRaLinear layer with a standard nn.Linear layer.
        This method is used to revert the LoRa modifications.
        """
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

        merged_weights = self.weight + (
            self.alpha * (self.lora_b.weight @ self.lora_a.weight)
        )
        linear.weight.data.copy_(merged_weights.data)

        if self.bias is not None:
            merged_bias = self.bias + (self.alpha * self.lora_b.bias)
            linear.bias.data.copy_(merged_bias.data)

        return linear

    def forward(self, input):
        """
        Forward pass for the LoRaLinear layer.
        This method applies the linear transformation with LoRa modifications.
        """
        # if we have a large input, merge the weights first before applying to the input
        batch_size = input.numel() / input.shape[-1]

        if (batch_size > self.out_features) or (batch_size > self.in_features):
            # large input, we merge weights first

            # No merging cost: BxRxIN + BxOUTxR + BxOUTxIN
            # Merging cost: OUTxIN + OUTxRxIN + BxOUTxIN
            # Better when OUTxIN + OUTxRxIN + BxOUTxIN < BxRxIN + BxOUTxR + BxOUTxIN
            #             OUTxIN + OUTxRxIN < BxRxIN + BxOUTxR
            merged_weights = self.weight + (
                self.alpha * (self.lora_b.weight @ self.lora_a.weight)
            )

            merged_bias = None
            if self.bias is not None:
                merged_bias = self.bias + (self.alpha * self.lora_b.bias)

            output = nn.functional.linear(input, merged_weights, merged_bias)

            return output
        else:
            # Apply the standard linear transformation
            base_output = nn.functional.linear(input, self.weight, self.bias)

            # Apply the LoRa modifications
            lora_output = self.lora_b(self.lora_a(input))

            return base_output + self.alpha * lora_output
