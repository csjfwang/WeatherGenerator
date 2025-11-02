# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import fnmatch
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

try:
    from peft import LoraConfig, get_peft_model, TaskType
    from peft.utils import _get_submodules
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT library not available. LoRA functionality will be disabled.")

from weathergen.utils.distributed import is_root


class WeatherGeneratorWrapper(nn.Module):
    """
    Wrapper class to make WeatherGenerator compatible with PEFT library.
    
    This wrapper handles the interface mismatch between PEFT and WeatherGenerator by
    providing a standard Hugging Face interface for PEFT
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._is_peft_call = False
        self._stored_args: Optional[Tuple[Any, ...]] = None
        self._stored_kwargs: Optional[Dict[str, Any]] = None
        
    def forward(self, *args, **kwargs):
        """
        Forward pass that handles both PEFT calls and normal training calls.
        For normal training calls, directly call the original model.
        For PEFT calls, also directly call the original model, let PEFT library handle adapter logic.
        """
        
        # Check if this is a PEFT call (only kwargs, no args)
        if len(args) == 0 and kwargs:

            model_params, batch, forecast_offset, forecast_steps, filtered_kwargs = self._translate_kwargs(kwargs)
            
            if model_params is None or batch is None:
                if self._stored_args is not None and len(self._stored_args) >= 2:
                    stored_model_params, stored_batch, stored_offset, stored_steps = self._stored_args[:4]
                    model_params = model_params or stored_model_params
                    batch = batch or stored_batch
                    forecast_offset = (
                        forecast_offset if forecast_offset is not None else stored_offset
                    )
                    forecast_steps = forecast_steps if forecast_steps is not None else stored_steps
                else:
                    raise RuntimeError(
                        "PEFT invocation could not be translated to WeatherGenerator forward signature. "
                        f"Received keys: {list(kwargs.keys())}"
                    )
            
            self._stored_args = (model_params, batch, forecast_offset, forecast_steps)
            self._stored_kwargs = filtered_kwargs
            
            return self.model.forward(model_params, batch, forecast_offset, forecast_steps, **filtered_kwargs)
        
        # Normal training call, store parameters for PEFT calls
        if len(args) >= 4:
            self._stored_args = args
            self._stored_kwargs = kwargs
        
        # Directly call the original model
        try:
            return self.model.forward(*args, **kwargs)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                # Filter out PEFT-specific arguments
                filtered_kwargs = {}
                peft_args = [
                    'input_ids', 'attention_mask', 'labels', 'output_attentions', 
                    'output_hidden_states', 'return_dict', 'inputs_embeds', 'past_key_values',
                    'use_cache', 'token_type_ids', 'position_ids', 'head_mask', 'cross_attn_head_mask'
                ]
                for key, value in kwargs.items():
                    if key not in peft_args:
                        filtered_kwargs[key] = value
                
                # Try again with filtered arguments
                return self.model.forward(*args, **filtered_kwargs)
            else:
                # Re-raise if it's a different type error
                raise
    
    def _translate_kwargs(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Dict[str, Any]]:
        """
        Translate PEFT/HuggingFace style keyword arguments to the WeatherGenerator signature.
        Returns positional arguments (model_params, batch, forecast_offset, forecast_steps)
        accompanied by remaining kwargs that should be forwarded as-is.
        """
        kw = dict(kwargs)  # work on a copy to avoid mutating the original dict
        
        model_params = kw.pop("model_params", None)
        batch = kw.pop("batch", None)
        forecast_offset = kw.pop("forecast_offset", None)
        forecast_steps = kw.pop("forecast_steps", None)
        
        #########################################################
        #    WJF: Need to clarify the mapping in another way    #
        #########################################################

        # Map Hugging Face style arguments (positional args got translated by PEFT)
        if model_params is None and "input_ids" in kw:
            model_params = kw.pop("input_ids")
        if batch is None and "attention_mask" in kw:
            batch = kw.pop("attention_mask")
        if forecast_offset is None and "inputs_embeds" in kw:
            forecast_offset = kw.pop("inputs_embeds")
        if forecast_steps is None and "labels" in kw:
            forecast_steps = kw.pop("labels")

        #########################################################
        
        # output_attentions can be either a bool (true HF argument) or our forecast_steps positional
        if forecast_steps is None and "output_attentions" in kw:
            candidate = kw.pop("output_attentions")
            if isinstance(candidate, bool) or candidate is None:
                # keep genuine HF argument
                kw["output_attentions"] = candidate
            else:
                forecast_steps = candidate
        
        # As a final fallback, try to recover from stored args if the translated values are clearly invalid
        if model_params is None or batch is None:
            return model_params, batch, forecast_offset, forecast_steps, {
                key: value for key, value in kw.items() if key not in self._peft_specific_args()
            }
        
        filtered_kwargs = {}
        for key, value in kw.items():
            if key not in self._peft_specific_args():
                filtered_kwargs[key] = value
        
        return model_params, batch, forecast_offset, forecast_steps, filtered_kwargs
    
    @staticmethod
    def _peft_specific_args() -> List[str]:
        return [
            "input_ids",
            "attention_mask",
            "labels",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "inputs_embeds",
            "past_key_values",
            "use_cache",
            "token_type_ids",
            "position_ids",
            "head_mask",
            "cross_attn_head_mask",
            "model_params",
            "batch",
            "forecast_offset",
            "forecast_steps",
        ]
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

_logger = logging.getLogger(__name__)


class AdapterManager:
    """
    Manages the attachment and configuration of parameter-efficient fine-tuning adapters
    to the WeatherGenerator model.
    
    This class provides a flexible way to attach LoRA adapters to specific modules
    in the model using regex pattern matching, while keeping the base model unchanged.
    """
    
    def __init__(self, model: nn.Module, config: DictConfig):
        """
        Initialize the AdapterManager.
        
        Args:
            model: The WeatherGenerator model to attach adapters to
            config: Configuration object containing PEFT settings
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required for adapter functionality")
            
        self.model = model
        self.config = config
        self.peft_model = None
        self.adapter_config = None
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate the PEFT configuration."""
        if not hasattr(self.config, 'peft'):
            raise ValueError("PEFT configuration not found in config")
            
        peft_config = self.config.peft
        if not peft_config.get('enabled', False):
            return
            
        if peft_config.get('method') != 'lora':
            raise ValueError("Only LoRA method is currently supported")
            
        if not peft_config.get('target_modules'):
            raise ValueError("No target modules specified for LoRA")
            
        if not peft_config.get('lora_config'):
            raise ValueError("LoRA configuration not provided")
    
    def attach_adapters(self) -> nn.Module:
        """
        Attach LoRA adapters to the model based on configuration.
        
        Returns:
            The model with LoRA adapters attached
        """
        if not self.config.peft.get('enabled', False):
            if is_root():
                _logger.info("PEFT is disabled, returning original model")
            return self.model
            
        if is_root():
            _logger.info("Attaching LoRA adapters to WeatherGenerator model")
            
        # Create LoRA configuration
        lora_config = self.config.peft.lora_config
        modules_to_save = lora_config.get('modules_to_save')
        if modules_to_save:
            # ensure modules_to_save is a list; allow comma-separated string in config
            if isinstance(modules_to_save, str):
                modules_to_save = [m.strip() for m in modules_to_save.split(",") if m.strip()]
            else:
                modules_to_save = list(modules_to_save)
        else:
            modules_to_save = None
        self.adapter_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # it tells PEFT “this isn’t a standard HF transformer task; just wrap it generically and don’t make language-model assumptions
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('alpha', 32),
            lora_dropout=lora_config.get('dropout', 0.1),
            bias=lora_config.get('bias', 'none'),
            target_modules=self._get_target_modules(),
            modules_to_save=modules_to_save,
        )
        
        # Apply LoRA using PEFT library with the wrapper
        try:
            # Wrap the model to make it compatible with PEFT
            wrapped_model = WeatherGeneratorWrapper(self.model)
            
            # Apply LoRA to the wrapped model
            self.peft_model = get_peft_model(wrapped_model, self.adapter_config)
            
            # Ensure any explicitly configured parameters stay trainable
            extra_params = self.config.peft.get('trainable_parameters', [])
            if extra_params:
                if isinstance(extra_params, str):
                    extra_params = [p.strip() for p in extra_params.split(",") if p.strip()]
                else:
                    extra_params = list(extra_params)
                param_items = list(self.peft_model.named_parameters())
                param_dict = dict(param_items)
                for pattern in extra_params:
                    matched = []
                    if pattern in param_dict:
                        matched = [param_dict[pattern]]
                        matched_names = [pattern]
                    else:
                        matched = [
                            param for name, param in param_items if fnmatch.fnmatchcase(name, pattern)
                        ]
                        matched_names = [
                            name for name, _ in param_items if fnmatch.fnmatchcase(name, pattern)
                        ]
                    if matched:
                        for param in matched:
                            param.requires_grad = True
                        if is_root():
                            _logger.debug(
                                "Set %s as trainable (matched pattern '%s')",
                                matched_names,
                                pattern,
                            )
                    elif is_root():
                        _logger.warning(
                            "Requested trainable parameter pattern '%s' not found in PEFT model",
                            pattern,
                        )
            
            if is_root():
                self._log_adapter_info()
                
            return self.peft_model
            
        except Exception as e:
            if is_root():
                _logger.error(f"Failed to apply PEFT adapters: {e}")
                _logger.info("Falling back to original model without adapters")
            return self.model
    
    def _get_target_modules(self) -> List[str]:
        """
        Get list of target module names that match the configured patterns.
        
        Returns:
            List of module names to apply LoRA to
        """
        target_modules = []
        target_patterns = self.config.peft.target_modules
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                for pattern in target_patterns:
                    if re.match(pattern, name):
                        target_modules.append(name)
                        if is_root():
                            _logger.debug(f"Matched module '{name}' with pattern '{pattern}'")
                        break
        
        if is_root():
            _logger.info(f"Found {len(target_modules)} modules for LoRA adaptation")
            _logger.debug(f"Target modules: {target_modules}")
            
        return target_modules
    
    def _log_adapter_info(self):
        """Log information about the attached adapters."""
        if self.peft_model is None:
            return
            
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        _logger.info(f"LoRA adapter attached successfully")
        _logger.info(f"Trainable parameters: {trainable_params:,}")
        _logger.info(f"Total parameters: {total_params:,}")
        _logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        # Log which modules have trainable parameters (only leaf LoRA modules)
        trainable_modules = []
        lora_modules = []
        
        for name, module in self.peft_model.named_modules():
            if any(p.requires_grad for p in module.parameters()):
                trainable_modules.append(name)
                # Check if this is a LoRA module (lora_A or lora_B, but not .default submodules)
                if ('lora_A' in name or 'lora_B' in name) and not name.endswith('.default'):
                    lora_modules.append(name)
        
        _logger.info(f"Total trainable modules: {len(trainable_modules)}")
        _logger.info(f"LoRA modules (lora_A/lora_B): {len(lora_modules)}")
        
        # Show LoRA modules specifically
        if lora_modules:
            _logger.info("LoRA modules:")
            for module_name in lora_modules:  # Show all LoRA modules
                _logger.info(f"  - {module_name}")
        
        # Show all parent modules for context
        # _logger.info("All parent modules (for context):")
        # parent_modules = [name for name in trainable_modules if 'lora_A' not in name and 'lora_B' not in name]
        # for module_name in parent_modules:
        #     _logger.info(f"  - {module_name}")
    
    def save_adapter(self, path: str, task_name: Optional[str] = None):
        """
        Save the LoRA adapter weights.
        
        Args:
            path: Path to save the adapter
            task_name: Optional task name for the adapter
        """
        if self.peft_model is None:
            raise RuntimeError("No adapter model available to save")
            
        if is_root():
            _logger.debug(f"Saving LoRA adapter to {path}")
            
        self.peft_model.save_pretrained(path)
        
        if task_name:
            # Save task-specific metadata
            import json
            metadata = {
                "task_name": task_name,
                "adapter_type": "lora",
                "config": self.adapter_config.to_dict()
            }
            with open(f"{path}/adapter_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
    
    def load_adapter(self, path: str):
        """
        Load LoRA adapter weights.
        
        Args:
            path: Path to load the adapter from
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required for adapter loading")
            
        if is_root():
            _logger.info(f"Loading LoRA adapter from {path}")
            
        # Load adapter configuration and weights
        from peft import PeftModel
        self.peft_model = PeftModel.from_pretrained(self.model, path)
        
        if is_root():
            _logger.info("LoRA adapter loaded successfully")
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of trainable parameters (only adapter parameters).
        
        Returns:
            List of trainable parameters
        """
        if self.peft_model is None:
            return []
            
        return [p for p in self.peft_model.parameters() if p.requires_grad]
    
    def disable_adapters(self):
        """Disable all adapters (set to inference mode)."""
        if self.peft_model is not None:
            self.peft_model.disable_adapters()
            if is_root():
                _logger.info("Adapters disabled")
    
    def enable_adapters(self):
        """Enable all adapters."""
        if self.peft_model is not None:
            self.peft_model.enable_adapters()
            if is_root():
                _logger.info("Adapters enabled")


def create_adapter_manager(model: nn.Module, config: DictConfig) -> AdapterManager:
    """
    Factory function to create an AdapterManager.
    
    Args:
        model: The WeatherGenerator model
        config: Configuration object
        
    Returns:
        AdapterManager instance
    """
    return AdapterManager(model, config)
