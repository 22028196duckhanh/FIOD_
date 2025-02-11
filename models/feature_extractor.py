from typing import List, Dict, Any
import torch
import torch.nn as nn

class FeatureExtractor:
    def __init__(self, model: nn.Module, num_layers: int = 8):
        #self.model = model.model  # backbone
        self.model = model
        self.feature_maps = {}
        self.hooks = []
        self._register_hooks(num_layers)

    def _hook_fn(self, name: str):
        def hook(module: nn.Module, input: Any, output: Any):
            self.feature_maps[name] = output

        return hook

    # def _register_hooks(self, num_layers: int):
    #     count = 0
    #     for name, layer in self.model.named_modules():
    #         if name.count('.') == 1 and count < num_layers:
    #             self.hooks.append(layer.register_forward_hook(self._hook_fn(name)))
    #             count += 1
    #         if count >= num_layers:
    #             break

    def _register_hooks(self, num_layers: int):
        count = 0
        target_layers = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
        for name, layer in self.model.named_modules():
            # Chỉ register hook cho các layer chính
            if name in target_layers and count < num_layers:
                self.hooks.append(layer.register_forward_hook(self._hook_fn(name)))
                count += 1
            if count >= num_layers:
                break

    def get_feature_maps(self, image: torch.Tensor) -> List[torch.Tensor]:
        self.feature_maps.clear()
        _ = self.model(image)
        return list(self.feature_maps.values())

    def __del__(self):
        for hook in self.hooks:
            hook.remove()