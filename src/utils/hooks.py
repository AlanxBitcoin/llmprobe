# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Design requirements (moved from PROJECT_DESIGN.md):
# - Low-level model operation API layer (extract/modify/query hooks).
# - Temporary model modifications should be restorable whenever possible.
# - Keep business orchestration outside this module.
"""
Hooks for direct manipulation of LLaMA models.
This module contains utilities for:
- Extracting hidden states from layers
- Skipping layers during forward pass
- Disabling attention heads
- Extracting FFN neuron activations
- Monitoring parameter changes
"""

import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict
from types import SimpleNamespace
import inspect

class HiddenStateHook:
    """Manager for extracting hidden states from model layers."""
    
    def __init__(self):
        self.hidden_states = defaultdict(dict)
        self.handles = []
    
    def _hook_fn(self, layer_name):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.hidden_states[layer_name]['output'] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    self.hidden_states[layer_name]['output'] = output[0].detach()
            if isinstance(input, tuple) and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    self.hidden_states[layer_name]['input'] = input[0].detach()
        return hook
    
    def register_hook(self, module, layer_name):
        """Register hook on a module."""
        handle = module.register_forward_hook(self._hook_fn(layer_name))
        self.handles.append(handle)
        return handle
    
    def get_hidden_states(self, layer_name):
        """Get hidden states for a specific layer."""
        return self.hidden_states.get(layer_name, {})
    
    def get_all_hidden_states(self):
        """Get all collected hidden states."""
        return dict(self.hidden_states)
    
    def clear(self):
        """Clear all hidden states."""
        self.hidden_states.clear()
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def rotate_half(x):
    """Rotates half the hidden dims of the input.
    
    Args:
        x: Input tensor of hidden states
        
    Returns:
        Tensor with rotated hidden dimensions
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    
    This function modifies the hidden states of query and key tensors by applying
    rotary position embeddings, which helps the model understand the relative
    positions of tokens in the sequence.

    Args:
        q (torch.Tensor): The query tensor with shape [batch, heads, seq_len, head_dim]
        k (torch.Tensor): The key tensor with shape [batch, heads, seq_len, head_dim]
        cos (torch.Tensor): The cosine part of the rotary embedding
        sin (torch.Tensor): The sine part of the rotary embedding
        unsqueeze_dim (int, *optional*, defaults to 1):
            The dimension along which to unsqueeze cos and sin for broadcasting.
            
    Returns:
        tuple(torch.Tensor): A tuple containing:
            - q_embed: Query tensor rotated using the Rotary Position Embedding
            - k_embed: Key tensor rotated using the Rotary Position Embedding
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value hidden states to match the number of attention heads.
    
    This function transforms hidden states from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim) by repeating along the head dimension.
    This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).

    Args:
        hidden_states (torch.Tensor): The hidden states to repeat
        n_rep (int): Number of times to repeat
        
    Returns:
        torch.Tensor: Repeated hidden states with shape 
                     (batch, num_key_value_heads * n_rep, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Forward pass for eager attention computation with hidden state processing.
    
    This function extracts and processes hidden states from query, key, and value tensors,
    computes attention weights, and applies attention to the value hidden states.
    
    Args:
        module: The attention module containing configuration
        query (torch.Tensor): Query hidden states [batch, num_heads, seq_len, head_dim]
        key (torch.Tensor): Key hidden states [batch, num_kv_heads, seq_len, head_dim]
        value (torch.Tensor): Value hidden states [batch, num_kv_heads, seq_len, head_dim]
        attention_mask (torch.Tensor | None): Optional attention mask
        scaling (float): Scaling factor for attention weights
        dropout (float): Dropout probability (default: 0.0)
        **kwargs: Additional arguments
        
    Returns:
        tuple: A tuple containing:
            - attn_output: Output attention hidden states
            - attn_weights: Computed attention weights
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def extract_hidden_states(model, layer_idx=None, detach=True):
    """Extracts hidden states from specific layers of the model.
    
    Args:
        model: The LLaMA model
        layer_idx (int, optional): Specific layer index to extract from.
                                   If None, extracts from all layers.
        detach (bool): Whether to detach the hidden states from the computation graph
        
    Returns:
        dict or list: Hidden states organized by layer
    """
    hidden_states = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                hidden_states[name] = output.detach() if detach else output
            elif hasattr(output, 'hidden_states') and output.hidden_states is not None:
                hidden_states[name] = output.hidden_states[-1].detach() if detach else output.hidden_states[-1]
        return hook
    
    return hidden_states, hook_fn


def modify_parameters(module, param_name, modification_fn):
    """Modifies model parameters using a custom modification function.
    
    Args:
        module: The model module
        param_name (str): Name of the parameter to modify
        modification_fn: Function that takes parameter and returns modified parameter
        
    Returns:
        bool: True if parameter was successfully modified, False otherwise
    """
    if hasattr(module, param_name):
        original_param = getattr(module, param_name)
        if isinstance(original_param, nn.Parameter):
            modified_param = modification_fn(original_param)
            setattr(module, param_name, nn.Parameter(modified_param))
            return True
    return False


def register_hidden_state_hook(module, hook_fn, hook_name="hidden_states"):
    """Registers a hook to monitor hidden states during forward pass.
    
    Args:
        module: The module to register hook on
        hook_fn: The hook function
        hook_name (str): Name identifier for the hook
        
    Returns:
        torch.utils.hooks.RemovableHandle: Handle to remove the hook later
    """
    return module.register_forward_hook(hook_fn)


def extract_layer_hidden_states(model, layer_indices: Union[int, List[int], None] = None, 
                                input_ids: Optional[torch.Tensor] = None,
                                **forward_kwargs) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Extract hidden states from specific layers.
    
    Args:
        model: The LLaMA model
        layer_indices: Layer index(es) to extract. If None, extract from all layers.
                      Can be int (single layer) or list of ints (multiple layers)
        input_ids: Input tensor for forward pass
        **forward_kwargs: Additional arguments for model forward pass
        
    Returns:
        Dictionary mapping layer indices to their hidden states
        
    Example:
        # Extract from layer 5
        states = extract_layer_hidden_states(model, layer_indices=5, input_ids=input_ids)
        
        # Extract from layers 0, 3, 5
        states = extract_layer_hidden_states(model, layer_indices=[0, 3, 5], input_ids=input_ids)
        
        # Extract from all layers
        states = extract_layer_hidden_states(model, input_ids=input_ids)
    """
    if not hasattr(model, 'layers'):
        raise ValueError("Model doesn't have 'layers' attribute")
    
    num_layers = len(model.layers)
    
    # Determine which layers to hook
    if layer_indices is None:
        target_layers = list(range(num_layers))
    elif isinstance(layer_indices, int):
        target_layers = [layer_indices]
    else:
        target_layers = list(layer_indices)
    
    hook_manager = HiddenStateHook()
    
    try:
        # Register hooks on target layers
        for layer_idx in target_layers:
            if 0 <= layer_idx < num_layers:
                hook_manager.register_hook(model.layers[layer_idx], f"layer_{layer_idx}")
        
        # Forward pass
        with torch.no_grad():
            if input_ids is not None:
                model(input_ids, **forward_kwargs)
            else:
                raise ValueError("input_ids is required for forward pass")
        
        # Collect results
        result = {}
        for layer_idx in target_layers:
            result[layer_idx] = hook_manager.get_hidden_states(f"layer_{layer_idx}")
        
        return result
    
    finally:
        hook_manager.remove_all_hooks()


def skip_layers(model, layer_indices: Union[int, List[int]], 
                input_ids: torch.Tensor, **forward_kwargs) -> torch.Tensor:
    """
    Skip one or more layers during forward pass by passing hidden states directly.
    
    Args:
        model: The LLaMA model
        layer_indices: Layer index(es) to skip. Can be int or list of ints
        input_ids: Input tensor
        **forward_kwargs: Additional arguments for model forward pass
        
    Returns:
        Output tensor with specified layers skipped
        
    Example:
        # Skip layer 3
        output = skip_layers(model, layer_indices=3, input_ids=input_ids)
        
        # Skip layers 2, 4, 6
        output = skip_layers(model, layer_indices=[2, 4, 6], input_ids=input_ids)
    """
    if not hasattr(model, 'layers'):
        raise ValueError("Model doesn't have 'layers' attribute")
    
    # Convert to list if single int
    if isinstance(layer_indices, int):
        skip_indices = [layer_indices]
    else:
        skip_indices = list(layer_indices)
    
    # Store original forward methods
    original_forwards = {}
    
    try:
        # Create wrapper for layers to skip
        for skip_idx in skip_indices:
            if 0 <= skip_idx < len(model.layers):
                original_forwards[skip_idx] = model.layers[skip_idx].forward
                
                def make_skip_forward(original_forward):
                    def skip_forward(hidden_states, **kwargs):
                        # Skip this layer, return input as is
                        return hidden_states
                    return skip_forward
                
                model.layers[skip_idx].forward = make_skip_forward(original_forwards[skip_idx])
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, **forward_kwargs)
        
        return outputs
    
    finally:
        # Restore original forward methods
        for skip_idx, original_forward in original_forwards.items():
            model.layers[skip_idx].forward = original_forward


class AttentionHeadDisabler:
    """Manages disabling attention heads in the model."""
    
    def __init__(self, model):
        self.model = model
        self.disabled_heads = defaultdict(set)
        self.handles = []
    
    def _create_head_mask_hook(self, layer_idx, disabled_heads):
        """Create a hook that masks out disabled attention heads."""
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 0:
                attn_output = output[0]
                if isinstance(attn_output, torch.Tensor):
                    # Create mask for disabled heads
                    batch_size, num_heads, seq_len, head_dim = attn_output.shape
                    mask = torch.ones(num_heads, device=attn_output.device)
                    for head_idx in disabled_heads:
                        if 0 <= head_idx < num_heads:
                            mask[head_idx] = 0
                    
                    # Apply mask
                    masked_output = attn_output * mask.view(1, -1, 1, 1)
                    return (masked_output,) + output[1:]
            return output
        return hook
    
    def disable_heads(self, head_indices: Union[int, List[int]], 
                     layer_indices: Union[int, List[int], None] = None):
        """
        Disable specific attention heads.
        
        Args:
            head_indices: Head index(es) to disable. Can be int or list of ints
            layer_indices: Layer(s) to apply to. If None, apply to all layers
            
        Example:
            # Disable head 0 in all layers
            disabler.disable_heads(head_indices=0)
            
            # Disable heads 1, 3 in layers 2, 5
            disabler.disable_heads(head_indices=[1, 3], layer_indices=[2, 5])
        """
        if not hasattr(self.model, 'layers'):
            raise ValueError("Model doesn't have 'layers' attribute")
        
        # Convert to lists if needed
        head_idx_list = [head_indices] if isinstance(head_indices, int) else list(head_indices)
        
        if layer_indices is None:
            layer_idx_list = list(range(len(self.model.layers)))
        elif isinstance(layer_indices, int):
            layer_idx_list = [layer_indices]
        else:
            layer_idx_list = list(layer_indices)
        
        # Register hooks for each layer
        for layer_idx in layer_idx_list:
            if 0 <= layer_idx < len(self.model.layers):
                for head_idx in head_idx_list:
                    self.disabled_heads[layer_idx].add(head_idx)
                
                # Register hook on attention module
                if hasattr(self.model.layers[layer_idx], 'self_attn'):
                    handle = self.model.layers[layer_idx].self_attn.register_forward_hook(
                        self._create_head_mask_hook(layer_idx, self.disabled_heads[layer_idx])
                    )
                    self.handles.append(handle)
    
    def enable_heads(self, head_indices: Union[int, List[int]], 
                    layer_indices: Union[int, List[int], None] = None):
        """Re-enable previously disabled heads."""
        head_idx_list = [head_indices] if isinstance(head_indices, int) else list(head_indices)
        
        if layer_indices is None:
            layer_idx_list = list(range(len(self.model.layers)))
        elif isinstance(layer_indices, int):
            layer_idx_list = [layer_indices]
        else:
            layer_idx_list = list(layer_indices)
        
        for layer_idx in layer_idx_list:
            for head_idx in head_idx_list:
                self.disabled_heads[layer_idx].discard(head_idx)
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def disable_attention_heads(model, head_indices: Union[int, List[int]], 
                           layer_indices: Union[int, List[int], None] = None) -> AttentionHeadDisabler:
    """
    Disable specific attention heads in model layers.
    
    Args:
        model: The LLaMA model
        head_indices: Head index(es) to disable
        layer_indices: Layer(s) to disable heads in. If None, applies to all layers
        
    Returns:
        AttentionHeadDisabler instance for managing head states
        
    Example:
        # Disable head 0 in all layers
        disabler = disable_attention_heads(model, head_indices=0)
        
        # Disable heads 1, 3 in layers 2, 5
        disabler = disable_attention_heads(model, head_indices=[1, 3], layer_indices=[2, 5])
    """
    disabler = AttentionHeadDisabler(model)
    disabler.disable_heads(head_indices, layer_indices)
    return disabler


def extract_ffn_neuron_hidden_states(model, layer_indices: Union[int, List[int], None] = None,
                                    neuron_indices: Union[int, List[int], None] = None,
                                    input_ids: Optional[torch.Tensor] = None,
                                    **forward_kwargs) -> Dict[int, torch.Tensor]:
    """
    Extract hidden states from FFN neurons in specific layers.
    
    Args:
        model: The LLaMA model
        layer_indices: Layer(s) to extract from. If None, extract from all layers
        neuron_indices: Specific neuron indices to extract. If None, extract all neurons
        input_ids: Input tensor for forward pass
        **forward_kwargs: Additional arguments for model forward pass
        
    Returns:
        Dictionary mapping layer indices to FFN neuron hidden states
        
    Example:
        # Extract all FFN neurons from layer 5
        states = extract_ffn_neuron_hidden_states(model, layer_indices=5, input_ids=input_ids)
        
        # Extract neurons 100-200 from all layers
        states = extract_ffn_neuron_hidden_states(model, neuron_indices=list(range(100, 200)), 
                                                 input_ids=input_ids)
    """
    if not hasattr(model, 'layers'):
        raise ValueError("Model doesn't have 'layers' attribute")
    
    num_layers = len(model.layers)
    
    # Determine which layers to hook
    if layer_indices is None:
        target_layers = list(range(num_layers))
    elif isinstance(layer_indices, int):
        target_layers = [layer_indices]
    else:
        target_layers = list(layer_indices)
    
    ffn_hidden_states = defaultdict(dict)
    handles = []
    
    def make_ffn_hook(layer_idx, neuron_idx_list):
        """Create hook function for FFN layer."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                tensor = output
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                tensor = output[0]
            else:
                return
            
            # Extract specific neurons or all
            if neuron_idx_list is None:
                ffn_hidden_states[layer_idx]['all_neurons'] = tensor.detach()
            else:
                # Extract specific neurons
                if tensor.dim() >= 2:
                    extracted = tensor[..., neuron_idx_list]
                    ffn_hidden_states[layer_idx]['neurons'] = extracted.detach()
        
        return hook
    
    try:
        # Register hooks on FFN modules
        for layer_idx in target_layers:
            if 0 <= layer_idx < num_layers and hasattr(model.layers[layer_idx], 'mlp'):
                # Hook on down_proj to get FFN output
                if hasattr(model.layers[layer_idx].mlp, 'down_proj'):
                    handle = model.layers[layer_idx].mlp.down_proj.register_forward_hook(
                        make_ffn_hook(layer_idx, neuron_indices)
                    )
                    handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            if input_ids is not None:
                model(input_ids, **forward_kwargs)
            else:
                raise ValueError("input_ids is required for forward pass")
        
        return dict(ffn_hidden_states)
    
    finally:
        for handle in handles:
            handle.remove()


class ParameterMonitor:
    """Monitor parameter changes during forward/backward pass."""
    
    def __init__(self, model):
        self.model = model
        self.param_snapshots = defaultdict(dict)
        self.handles = []
    
    def _create_param_hook(self, param_name):
        """Create hook to monitor parameter changes."""
        def hook(grad):
            # Store gradient information
            self.param_snapshots[param_name]['grad'] = grad.detach().clone() if grad is not None else None
            return grad
        return hook
    
    def register_parameter_monitor(self, layer_idx: int, neuron_idx: Optional[int] = None,
                                   param_type: str = 'weight'):
        """
        Register monitoring on specific neuron parameters.
        
        Args:
            layer_idx: Layer index
            neuron_idx: Specific neuron index. If None, monitors whole layer
            param_type: Type of parameter ('weight', 'bias')
        """
        if not hasattr(self.model, 'layers') or layer_idx >= len(self.model.layers):
            return False
        
        layer = self.model.layers[layer_idx]
        param_key = f"layer_{layer_idx}_{param_type}_neuron_{neuron_idx}" if neuron_idx else f"layer_{layer_idx}_{param_type}"
        
        # Monitor attention weights
        if hasattr(layer, 'self_attn'):
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, proj_name):
                    module = getattr(layer.self_attn, proj_name)
                    if hasattr(module, param_type):
                        param = getattr(module, param_type)
                        if param is not None and hasattr(param, 'register_hook'):
                            handle = param.register_hook(
                                self._create_param_hook(f"{param_key}_{proj_name}")
                            )
                            self.handles.append(handle)
        
        # Monitor MLP weights
        if hasattr(layer, 'mlp'):
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(layer.mlp, proj_name):
                    module = getattr(layer.mlp, proj_name)
                    if hasattr(module, param_type):
                        param = getattr(module, param_type)
                        if param is not None and hasattr(param, 'register_hook'):
                            handle = param.register_hook(
                                self._create_param_hook(f"{param_key}_{proj_name}")
                            )
                            self.handles.append(handle)
        
        return True
    
    def get_parameter_snapshot(self, layer_idx: int, param_type: str = 'weight') -> Dict:
        """Get snapshot of layer parameters."""
        snapshots = {}
        if not hasattr(self.model, 'layers') or layer_idx >= len(self.model.layers):
            return snapshots
        
        layer = self.model.layers[layer_idx]
        
        # Collect attention parameters
        if hasattr(layer, 'self_attn'):
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, proj_name):
                    module = getattr(layer.self_attn, proj_name)
                    if hasattr(module, param_type):
                        param = getattr(module, param_type)
                        if param is not None:
                            snapshots[f"attn_{proj_name}"] = param.detach().clone()
        
        # Collect MLP parameters
        if hasattr(layer, 'mlp'):
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(layer.mlp, proj_name):
                    module = getattr(layer.mlp, proj_name)
                    if hasattr(module, param_type):
                        param = getattr(module, param_type)
                        if param is not None:
                            snapshots[f"mlp_{proj_name}"] = param.detach().clone()
        
        return snapshots
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def extract_neuron_parameters(model, layer_idx: int, neuron_idx: Optional[int] = None,
                             param_type: str = 'weight') -> Dict[str, torch.Tensor]:
    """
    Extract parameters (weights/biases) from specific neurons or layers.
    
    Args:
        model: The LLaMA model
        layer_idx: Layer index
        neuron_idx: Specific neuron index. If None, extract whole layer
        param_type: Type of parameter ('weight', 'bias')
        
    Returns:
        Dictionary of parameter tensors
        
    Example:
        # Extract all weights from layer 5
        params = extract_neuron_parameters(model, layer_idx=5, param_type='weight')
        
        # Extract bias from specific neuron
        params = extract_neuron_parameters(model, layer_idx=3, neuron_idx=100, param_type='bias')
    """
    monitor = ParameterMonitor(model)
    params = monitor.get_parameter_snapshot(layer_idx, param_type)
    monitor.remove_all_hooks()
    return params


def compare_neuron_parameters(model, layer_idx: int, neuron_idx: Optional[int] = None,
                             before_fn: Optional[Callable] = None,
                             after_fn: Optional[Callable] = None) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compare neuron parameters before and after some operation.
    
    Args:
        model: The LLaMA model
        layer_idx: Layer index
        neuron_idx: Specific neuron index
        before_fn: Function to call before the operation
        after_fn: Function to call after the operation
        
    Returns:
        Dictionary with 'before' and 'after' parameter snapshots
        
    Example:
        def before_op():
            # Some operation
            pass
        
        def after_op():
            # Some operation
            pass
        
        result = compare_neuron_parameters(model, layer_idx=5, 
                                          before_fn=before_op, after_fn=after_op)
        print("Weight change:", result['after']['mlp_down_proj'] - result['before']['mlp_down_proj'])
    """
    result = {}
    
    # Get before parameters
    monitor = ParameterMonitor(model)
    result['before'] = monitor.get_parameter_snapshot(layer_idx)
    monitor.remove_all_hooks()
    
    # Run operations if provided
    if before_fn is not None:
        before_fn()
    
    # Run after operation
    if after_fn is not None:
        after_fn()
    
    # Get after parameters
    monitor = ParameterMonitor(model)
    result['after'] = monitor.get_parameter_snapshot(layer_idx)
    monitor.remove_all_hooks()
    
    return result


def starting_from_middle_layer(
    model,
    *,
    start_layer_idx: int,
    hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_hidden_states: bool = True,
    return_dict: bool = True,
    **forward_kwargs,
) -> Dict[str, Any]:
    """
    Continue forward by directly running remaining layers from a middle hidden state.

    Practical implementation:
    - Treat `hidden_state` as the output of `start_layer_idx`.
    - Directly run layers (start_layer_idx + 1 .. end) in order.
    - No full-model re-run, and no front-half attention recomputation.

    Args:
        model: Causal LM model (e.g., LlamaForCausalLM).
        start_layer_idx: Decoder layer index to inject hidden state into.
        hidden_state: Replacement hidden state. Supported shapes:
            [hidden_dim], [seq_len, hidden_dim], [batch, seq_len, hidden_dim].
        input_ids: Input token ids [batch, seq_len] used for the forward call.
        attention_mask: Optional attention mask.
        output_hidden_states: Whether to include hidden states in output.
        return_dict: Whether model forward returns a dict-like object.
        **forward_kwargs: Extra kwargs passed to model forward.

    Returns:
        Dict with keys:
            - outputs: raw model outputs
            - logits: output logits tensor (if available)
            - remaining_hidden_states: hidden states after the injected layer
            - start_layer_idx: injected layer index
    """
    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose decoder layers via `model.layers` or `layers`.")

    num_layers = len(layers)
    if not (0 <= int(start_layer_idx) < num_layers):
        raise ValueError(f"start_layer_idx out of range: {start_layer_idx}, num_layers={num_layers}")

    current_hidden = _coerce_hidden_tensor(
        hidden_state=hidden_state,
        input_ids=input_ids,
        model=model,
    )
    collected_hidden_states: list[torch.Tensor] = [current_hidden]

    with torch.no_grad():
        batch_size = int(current_hidden.shape[0])
        seq_len = int(current_hidden.shape[1])
        device = current_hidden.device
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
        rotary = getattr(base_model, "rotary_emb", None) if base_model is not None else None
        position_embeddings = None
        if rotary is not None:
            try:
                position_embeddings = rotary(current_hidden, position_ids)
            except TypeError:
                position_embeddings = rotary(current_hidden, position_ids=position_ids)

        for layer_idx in range(int(start_layer_idx) + 1, num_layers):
            layer = layers[layer_idx]
            candidate_kwargs = {
                "hidden_states": current_hidden,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_value": None,
                "output_attentions": False,
                "use_cache": False,
                "cache_position": cache_position,
                "position_embeddings": position_embeddings,
            }
            sig = inspect.signature(layer.forward)
            accepted = set(sig.parameters.keys())
            layer_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
            layer_out = layer(**layer_kwargs)

            if isinstance(layer_out, tuple):
                current_hidden = layer_out[0]
            else:
                current_hidden = layer_out
            if not isinstance(current_hidden, torch.Tensor):
                raise ValueError(f"Layer {layer_idx} did not return hidden states tensor")
            collected_hidden_states.append(current_hidden)

        # Llama-style final norm before lm_head.
        base_model_norm = getattr(base_model, "norm", None) if base_model is not None else None
        hidden_for_logits = base_model_norm(current_hidden) if base_model_norm is not None else current_hidden

        lm_head = getattr(model, "lm_head", None)
        logits = lm_head(hidden_for_logits) if lm_head is not None else None

    # Keep output hidden-state semantics aligned with HF model outputs:
    # final hidden state should be after final norm.
    if collected_hidden_states:
        collected_hidden_states[-1] = hidden_for_logits
    remaining = tuple(collected_hidden_states[1:]) if len(collected_hidden_states) > 1 else tuple()
    outputs = SimpleNamespace(
        hidden_states=tuple(collected_hidden_states),
        logits=logits,
    )

    return {
        "outputs": outputs,
        "logits": logits,
        "remaining_hidden_states": remaining,
        "start_layer_idx": int(start_layer_idx),
    }


def _coerce_hidden_tensor(hidden_state: torch.Tensor, input_ids: torch.Tensor, model) -> torch.Tensor:
    """Normalize hidden state to [B, S, D] on model device/dtype."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    replacement = hidden_state.to(device=device, dtype=dtype)

    if replacement.ndim == 3:
        return replacement
    if replacement.ndim == 2:
        return replacement.unsqueeze(0)
    if replacement.ndim == 1:
        # Last-token vector only; build a single-step sequence.
        return replacement.view(1, 1, -1)
    raise ValueError(f"Unsupported hidden_state ndim={replacement.ndim}, expected 1/2/3")


def _patch_hidden_tensor(current_hidden: torch.Tensor, replacement_hidden: torch.Tensor) -> torch.Tensor:
    """Patch current hidden tensor [B, S, D] with replacement hidden state."""
    replacement = replacement_hidden.to(device=current_hidden.device, dtype=current_hidden.dtype)
    patched = current_hidden.clone()

    if replacement.ndim == 1:
        if replacement.shape[0] != patched.shape[-1]:
            raise ValueError(
                f"Hidden size mismatch for 1D replacement: expected {patched.shape[-1]}, got {replacement.shape[0]}"
            )
        patched[:, -1, :] = replacement.unsqueeze(0)
        return patched

    if replacement.ndim == 2:
        if replacement.shape[-1] != patched.shape[-1]:
            raise ValueError(
                f"Hidden size mismatch for 2D replacement: expected {patched.shape[-1]}, got {replacement.shape[-1]}"
            )
        seq_len = min(patched.shape[1], replacement.shape[0])
        patched[:, :seq_len, :] = replacement[:seq_len, :].unsqueeze(0)
        return patched

    if replacement.ndim == 3:
        if replacement.shape[-1] != patched.shape[-1]:
            raise ValueError(
                f"Hidden size mismatch for 3D replacement: expected {patched.shape[-1]}, got {replacement.shape[-1]}"
            )
        batch = min(patched.shape[0], replacement.shape[0])
        seq_len = min(patched.shape[1], replacement.shape[1])
        patched[:batch, :seq_len, :] = replacement[:batch, :seq_len, :]
        return patched

    raise ValueError(f"Unsupported replacement hidden_state ndim={replacement.ndim}, expected 1/2/3")
