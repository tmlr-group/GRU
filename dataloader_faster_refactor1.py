import torch
from torch import nn
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available
import datasets
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.nn.functional as F
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss 
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
import csv 
import pickle
import math
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import sys

#+
from transformers.utils import is_sagemaker_mp_enabled
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

# for solve qp
import quadprog
from quadprog import solve_qp
import gc




from typing import Union, Iterable

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def get_total_norm(parameters: _tensor_or_tensors)-> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    
    norm_type=2.0
    
    if len(grads) == 0:
        return torch.tensor(0.0)
    
    norms = [torch.linalg.vector_norm(g, norm_type) for g in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
    return total_norm




def print_grad_info_ds(model):
    for name, param in model.named_parameters():
        grad_data = deepspeed.utils.safe_get_full_grad(param) # for deepspeed
        if grad_data is not None:
            print(f"Grad info - Layer: {name}, Gradient Norm: {grad_data}")
        else:
            print(f"Grad info - Layer: {name}, No gradients")


def print_metrics(step, metrics):
    """
    Print current training metrics in a formatted way.

    Args:
        epoch (int): Current epoch or iteration number.
        metrics (dict): A dictionary containing metric names as keys and their current values.
    """
    # Prepare the formatted string
    metrics_string = ', '.join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    print(f"Step {step}: {metrics_string}")

def monitor_gradients(model):
    """
    Monitors and prints the gradient status for each layer of the model.

    Args:
        model (torch.nn.Module): The model to monitor.
    """
    has_grad = []
    zero_grad = []
    none_grad = []
    wo_grad = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                if torch.any(param.grad.abs() > 0):
                    has_grad.append(name)
                else:
                    zero_grad.append(name)
            else:
                none_grad.append(name)
        else:
            wo_grad.append(name)


    print("Layers with non-zero gradients:")
    for name in has_grad:
        print(f"  {name}: Gradient has non-zero values.")

    print("\nLayers with zero gradients:")
    for name in zero_grad:
        print(f"  {name}: Gradient is zero.")

    print("\nLayers with no gradients:")
    for name in none_grad:
        print(f"  {name}: No gradient computed.")


    print("\nLayers without gradients:")
    for name in wo_grad:
        print(f"  {name}: wo gradient.")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
    
class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        #print(kwargs)
        self.gradient_accumulation_steps = kwargs.pop('gradient_accumulation_steps')
        self.model_name = kwargs.pop('model_name')
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.seed = kwargs.pop('seed')
        self.gru_type = kwargs.pop('gru_type')
        print("gru_type",self.gru_type)
        self.ds = kwargs.pop('deepspeed')
        print("deepspeed:",self.ds)
        self.gradient_edit = False 
        self.ref_policy = kwargs.pop('ref_policy')
        self.beta = kwargs.pop('beta')
        self.ce_coeff = kwargs.pop('ce_coeff')
        self.moving_avg = kwargs.pop('moving_avg')

        self.cos_clip = kwargs.pop('cos_clip')

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

        # Determine if gradient editing is needed based on the loss type suffix
        if self.loss_type.endswith('_gru') or self.loss_type.endswith('_gru+KL')or self.loss_type.endswith('_gru+ce'):
            self.gradient_edit = True
            print("Executing gradient editing...")

        if 'npo' in self.loss_type or 'KL' in self.loss_type or 'gd_gru' in self.loss_type:
            if self.ds:
                self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)
            else:
                self.oracle_model = self.oracle_model.to(self.args.device)

        self.dotp_retain = None

        self.flattened_gradient = 0
        self.flattened_memory = 0
        self.flattened_memory_old = 0.0
        self.flattened_memory_accumulation = 0.0

        self.structure_map = None

        self.steps = 0

        self.check_retain_loss = True

        self.with_KL_gradient = False
        self.KL_flattened_gradient = 0
        if self.loss_type.endswith('_gru+KL'):
            self.with_KL_gradient = True
            print("adding KL gradient.......")
            self.KL_coef = 1.0
            
            print("KL coef:",self.KL_coef)

        self.with_ce_gradient = False
        if self.loss_type.endswith('_gru+ce'):
            self.with_ce_gradient = True
            print("adding NLL gradient.......")


        self.gradient_accum = {}
                        
        self.memory_grad = {}

        self.KL_gradient_accum = {}

        print("moving avg is:",self.moving_avg)


        print("ce_coeff:",self.ce_coeff)



    
    def get_norm(self):
        forget_norm = torch.norm(self.flattened_gradient, p=2).item()
        retain_norm = torch.norm(self.flattened_memory, p=2).item()
        print("forget_norm:",forget_norm," ","retain_norm:",retain_norm)

    def orthogonal_component(self, g, g1):

        self.get_norm()
        g1g1 = self.compute_total_gradient_dot_product(g1, self.structure_map, g1, self.structure_map)
        gg1 = self.dotp_retain
        print(gg1/g1g1)
        projection = gg1/g1g1* g1
        orthogonal = g - projection

        return orthogonal

    def orthogonal_component_precise(self, g, g1):
        self.get_norm()
        
        dtype = g.dtype
        shape = g.shape

        g = g.cpu().to(torch.float64).numpy()  # Convert to float64 before to NumPy
        g1 = g1.cpu().to(torch.float64).numpy()


        g = np.asarray(g).flatten()
        g1 = np.asarray(g1).flatten()


        projection = np.dot(g, g1) / np.dot(g1, g1) * g1

        orthogonal = g - projection


        scale_factor = 1


        return torch.from_numpy(orthogonal).to(dtype).view(shape) * scale_factor


    def orthogonal_component_COSCLIP(self, g, g1):
        # Compute norms
        retain_norm = torch.norm(g1, p=2).item()
        original_norm = torch.norm(g, p=2).item()

        # Compute original cos(theta)
        # dotp_retain = <g, g1>
        gg1 = self.dotp_retain  
        cos_theta = gg1 / (original_norm * retain_norm)

        # Cos clipping
        c = -0.5 #-0.2
        if cos_theta < c:
            print("cos_theta before clip:", cos_theta,"c:",c)
            cos_theta = c

       
        factor = (original_norm * cos_theta) / retain_norm
        ddot_g_u = g - factor * g1

        # Print debug info
        ddot_norm = torch.norm(ddot_g_u, p=2).item()
        print("forget_norm:", original_norm, "retain_norm:", retain_norm, "ddot_norm:", ddot_norm)

        return ddot_g_u


    
    def store_grads(self, model, loss=None, typ=None):
        """
        Accumulates gradients of specified layers, preserving their original shapes.
        Optionally, adjusts which layers are trainable just before computing gradients.

        Args:
            model (torch.nn.Module): The model from which to store gradients.
            loss (torch.Tensor, optional): The loss tensor to perform backward operation. If provided, will compute gradients.

        Returns:
            None: Modifies internal tensors to store accumulated gradients.
        """

        # Perform backward pass if a loss tensor is provided
        if loss:
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

        # Loop through parameters and accumulate gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

                # Choose the correct dictionary based on 'typ'
                if typ == "objective":
                    target_dict = self.gradient_accum
                elif typ == "retain":
                    target_dict = self.memory_grad
                elif typ == "KL_objective":
                    target_dict = self.KL_gradient_accum
                else:
                    raise ValueError("Invalid type specified for gradient storage")

                # Initialize the dictionary key if it doesn't exist
                if name not in target_dict:
                    target_dict[name] = torch.zeros_like(param.grad, device=param.grad.device)  # Initialize on the same device

                # Accumulate the gradients
                target_dict[name] += param.grad.detach()

        if loss:
            model.zero_grad()

        
    def flatten_and_store_grads(self):
        """
        Flattens accumulated gradients from different gradient dictionaries, moves them to CPU,
        and stores them along with a structure map for each type of gradient.
        """

        # Helper function to flatten gradients, move to CPU, and record their original structure
        def flatten_to_cpu_and_record_structure(gradient_dict):
            flattened_grads = []
            structure_map = []
            for name, grad in gradient_dict.items():
                if grad is not None:
                    grad_flat = grad.view(-1)
                    flattened_grads.append(grad_flat)
                    structure_map.append((name, grad.shape))

            if flattened_grads:
                return torch.cat(flattened_grads).to('cpu'), structure_map
            else:
                return torch.tensor([], dtype=torch.float32).to('cpu'), []

  
        self.flattened_gradient, self.structure_map = flatten_to_cpu_and_record_structure(self.gradient_accum)
      
        self.flattened_memory_accumulation, _ = flatten_to_cpu_and_record_structure(self.memory_grad)
       
        self.KL_flattened_gradient, _ = flatten_to_cpu_and_record_structure(self.KL_gradient_accum)
      
        
    
    
    def compute_total_gradient_dot_product(self, flattened_grads1, structure_map1, flattened_grads2, structure_map2):
        """
        Computes the total dot product between gradients from two sets of flattened gradients and their respective structure maps.

        Args:
            flattened_grads1 (torch.Tensor): The first flattened gradient tensor.
            structure_map1 (list): A list of tuples containing parameter names and their corresponding shapes for the first set of gradients.
            flattened_grads2 (torch.Tensor): The second flattened gradient tensor.
            structure_map2 (list): A list of tuples containing parameter names and their corresponding shapes for the second set of gradients.

        Returns:
            float: The total dot product summed across all matching layers.
        """
        #assert len(structure_map1) == len(structure_map2), "Both gradient structures must contain the same number of elements."

        total_dot_product = 0.0
        index = 0

        # Ensure both gradient tensors are on the same device
        flattened_grads1 = flattened_grads1.to('cuda')
        flattened_grads2 = flattened_grads2.to('cuda')

        # for ((name1, shape1), (name2, shape2)) in zip(structure_map1, structure_map2):
        #     assert name1 == name2 and shape1 == shape2, f"Gradient mismatch: {name1} vs {name2} or {shape1} vs {shape2}"


        for ((name1, shape1), (name2, shape2)) in zip(structure_map1, structure_map2):
            assert shape1 == shape2, f"Gradient mismatch: {name1} vs {name2} or {shape1} vs {shape2}"


            size = np.prod(shape1)  # Total number of elements in this layer's gradient
            grad_slice1 = flattened_grads1[index:index + size].view(shape1)
            grad_slice2 = flattened_grads2[index:index + size].view(shape2)

            # Compute the dot product of the two gradient slices
            dot_product = (grad_slice1 * grad_slice2).sum()
            total_dot_product += dot_product.item()

            index += size

        return total_dot_product

    def restore_gradients_from_flat(self, model):
        """
        Restores gradients to the model's parameters directly from a flattened gradient tensor.

        Args:
            model (torch.nn.Module): The model to which the gradients will be restored.
            flattened_grads (torch.Tensor): The flattened gradient tensor.
            structure_map (list): A list of tuples containing parameter names and their corresponding shapes.
        """

        index = 0  # Index to track position in the flattened gradient tensor

        for name, shape in self.structure_map:
            size = np.prod(shape)  # Total number of elements in this gradient
            if size == 0:  # Skip layers with no parameters
                continue

            # Extract the relevant slice from the flattened gradient tensor
            grad_slice = self.flattened_gradient[index:index + size].view(shape)

            # Find the corresponding parameter in the model
            param = next((p for n, p in model.named_parameters() if n == name), None)
            if param.requires_grad:
                # Check if the shape of the extracted gradient matches the parameter's shape
                if grad_slice.shape != param.shape:
                    raise ValueError(f"Gradient shape mismatch for {name}: expected {param.shape}, got {grad_slice.shape}")

                # Set the parameter's gradient to the extracted slice
                param.grad = grad_slice.to(param.device)

            index += size  # Update index to the start of the next gradient slice

        if index != self.flattened_gradient.numel():
            raise ValueError("Total number of gradient elements does not match the length of the flattened gradient tensor.")

    def pipeline(self, model):
        '''
        There are three constraint modes: retain constraint, unlearn constraint, pareto constraint
        '''

        if self.dotp_retain<0:
            print("dotp_retain:",self.dotp_retain)

            if self.gru_type == "proj":
                if self.cos_clip:
                    self.flattened_gradient = self.orthogonal_component_COSCLIP(self.flattened_gradient, self.flattened_memory)
                else:
                    #self.flattened_gradient = self.orthogonal_component(self.flattened_gradient, self.flattened_memory)
                    self.flattened_gradient = self.orthogonal_component_precise(self.flattened_gradient, self.flattened_memory)
                    

            torch.cuda.empty_cache()  


            
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], extra_arg=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

        else:
            # overwrite grad
            if self.gradient_edit:


                if self.steps % self.gradient_accumulation_steps == 0:

                    # Flatten and move accumulated gradients to CPU before clearing
                    self.flatten_and_store_grads()

                    self.gradient_accum = {}
                        
                    self.memory_grad = {}

                    self.KL_gradient_accum = {}

                    self.flattened_memory = self.moving_avg * self.flattened_memory_accumulation + (1 - self.moving_avg) * self.flattened_memory_old
                    self.flattened_memory_old = self.flattened_memory
                    self.dotp_retain = self.compute_total_gradient_dot_product(self.flattened_gradient, self.structure_map, 
                                                                        self.flattened_memory, self.structure_map)
                    self.pipeline(model)

                    if self.with_KL_gradient:
                        self.flattened_gradient += self.KL_flattened_gradient * self.KL_coef
                        self.KL_flattened_gradient = 0

                    if self.with_ce_gradient:
                        self.flattened_gradient += self.flattened_memory_accumulation * self.ce_coeff

                    self.restore_gradients_from_flat(model)
                    self.flattened_memory_accumulation = 0

            else:
                self.accelerator.backward(loss, **kwargs) 
                #monitor_gradients(model)
        
        #print_grad_info_ds(model)
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def get_train_dataloader(self):
        """
        Override the original get_train_dataloader function simply for debugging.
        This is identical to the get_train_dataloader function in transformer.Trainer.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.state.global_step)
        print(f'Generator........Epoch-{self.state.global_step}')

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            
            dataloader_params["generator"] = generator
            dataloader_params["shuffle"] = True # set shuffle=True with specified generator.
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )
        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def compute_loss(self, model, inputs, return_outputs=False):

        if self.loss_type =="ga":

            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)     
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss
            retain_loss = 0
            if self.check_retain_loss:
                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                with torch.no_grad():
                    retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
                retain_loss = retain_outputs.loss
                del retain_outputs

            total_norm = 0
            total_norm = get_total_norm(model.parameters())

            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss,
                "unlearn total_norm":total_norm

            }
            print_metrics(self.steps, metrics)

        elif self.loss_type == "simnpo":
            gamma = 0.0
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss_mask = labels != -100
            forget_loss = get_batch_loss(outputs.logits, labels) / loss_mask.sum(-1) - gamma

            loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

            self.steps +=1
            metrics = {
                'Loss': loss,

            }
            print_metrics(self.steps, metrics)



        elif self.loss_type == "simnpo_gru+ce":
            gamma = 0.0
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss_mask = labels != -100
            forget_loss = get_batch_loss(outputs.logits, labels) / loss_mask.sum(-1) - gamma

            loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

            del outputs
            self.store_grads(model, loss=loss, typ = "objective")


            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            del retain_outputs
            self.store_grads(model, loss=retain_loss, typ = "retain")


            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss

            }
            print_metrics(self.steps, metrics)

        elif self.loss_type == 'simnpo_gd':
            gamma = 0.0
            npo_coeff = 1.0
            grad_diff_coeff = 0.1
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss_mask = labels != -100
            forget_loss = get_batch_loss(outputs.logits, labels) / loss_mask.sum(-1) - gamma
            forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = npo_coeff * forget_loss + grad_diff_coeff * retain_loss

            self.steps +=1
            metrics = {
                'forget_loss': forget_loss,
                'Loss': loss,
                'retain_loss': retain_loss

            }
            print_metrics(self.steps, metrics)



        elif self.loss_type == 'simnpo_gd_gru':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss_mask = labels != -100
            forget_loss = get_batch_loss(outputs.logits, labels) / loss_mask.sum(-1) 
            forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss +  retain_loss * 0.1
            del outputs
            self.store_grads(model, loss=loss, typ = "objective")


            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            del retain_outputs
            self.store_grads(model, loss=retain_loss, typ = "retain")

            self.steps +=1
            metrics = {
                'forget_loss': forget_loss,
                'Loss': loss,
                'retain_loss': retain_loss

            }
            print_metrics(self.steps, metrics)


        elif self.loss_type == "idk":

            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss


            self.steps +=1
            metrics = {
                'Loss': loss

            }
            print_metrics(self.steps, metrics)


        elif self.loss_type == "idk_gru":

            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

            del outputs
            self.store_grads(model, loss=loss, typ = "objective")

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            del retain_outputs
            self.store_grads(model, loss=retain_loss, typ = "retain")


            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss

            }
            print_metrics(self.steps, metrics)

        elif self.loss_type == "gd_gru":
                gd_coef = 1.0
                forget_inputs, retain_inputs = inputs
                input_ids, labels, attention_mask = forget_inputs
                outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_loss = outputs.loss
                forget_loss = forget_loss * -1

                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                retain_loss = retain_outputs.loss
                loss = forget_loss + gd_coef * retain_loss
                del outputs
                self.store_grads(model, loss=loss, typ = "objective")

                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
                retain_loss = retain_outputs.loss
                del retain_outputs
                self.store_grads(model, loss=retain_loss, typ = "retain")


                self.steps +=1
                metrics = {
                    'Loss': loss,
                    'retain_loss': retain_loss

                }
                print_metrics(self.steps, metrics)



        
        
        elif self.loss_type == 'npo_gd':

            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss

            loss += retain_loss

                   
            total_norm = 0
            total_norm = get_total_norm(model.parameters())

            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss,
                "unlearn total_norm":total_norm

            }
            print_metrics(self.steps, metrics)



        elif self.loss_type == 'npo_gd_gru':


            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss

            loss += retain_loss



            self.store_grads(model, loss=loss, typ = "objective")

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            del retain_outputs
            self.store_grads(model, loss=retain_loss, typ = "retain")

            total_norm = 0
            total_norm = get_total_norm(model.parameters())

            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss,
                "unlearn total_norm":total_norm

            }
            print_metrics(self.steps, metrics)
                        
                
        elif self.loss_type == "gd":
            gd_coef = 1
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + gd_coef * retain_loss

            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss

            }
            print_metrics(self.steps, metrics)

        elif self.loss_type == "ga_KL":
            KL_coef = 5.0

            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss += retain_loss*KL_coef

            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss
            }
            print_metrics(self.steps, metrics)
 


        elif self.loss_type == "ga_KL_gru":

            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss += retain_loss * 5.0

            del outputs
            self.store_grads(model, loss=loss, typ = "objective")

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            del retain_outputs
            self.store_grads(model, loss=retain_loss, typ = "retain")


            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss
            }
            print_metrics(self.steps, metrics)
        

        elif self.loss_type in ["ga_gru","ga_gru+KL","ga_gru+ce"]:

            #monitor_gradients(model)
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs

            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)        
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss
            del outputs
            self.store_grads(model, loss=forget_loss, typ = "objective")

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            del retain_outputs
            self.store_grads(model, loss=retain_loss, typ = "retain")

                                            
            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss,
                'forget_loss': forget_loss
            }
            print_metrics(self.steps, metrics)
   
            del retain_loss
            del forget_loss

        elif self.loss_type in ['npo',"npo_KL"]:

            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

            retain_loss = 0
            if self.loss_type == "npo_KL":
                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                with torch.no_grad():
                    retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
                retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

                current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
                current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

                #minimum KL divergence
                retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
                loss += retain_loss
        
            total_norm = 0
            total_norm = get_total_norm(model.parameters())

            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss,
                "unlearn total_norm":total_norm

            }
            print_metrics(self.steps, metrics)




        elif self.loss_type in ["npo_KL_gru"]:

            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

            retain_loss = 0
            
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss += retain_loss



            self.store_grads(model, loss=loss, typ = "objective")

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            del retain_outputs
            self.store_grads(model, loss=retain_loss, typ = "retain")


            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss
            }
            print_metrics(self.steps, metrics)


        elif self.loss_type in ["npo_gru","npo_gru+KL"]:
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs

            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 

            del outputs
            del forget_loss_current
            
            objective_loss = loss
            self.store_grads(model, loss=objective_loss, typ = "objective")

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            del retain_outputs
            self.store_grads(model, loss=retain_loss, typ = "retain")

            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss,
            }
            print_metrics(self.steps, metrics)
  

        elif self.loss_type == 'ga_grukl':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs

            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)        
            forget_loss = outputs.loss
            loss = forget_loss * -1
            del outputs
            objective_loss = loss
            self.store_grads(model, loss=objective_loss, typ = "objective")


            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])
            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])
            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            del retain_outputs
            self.store_grads(model, loss=retain_loss, typ = "retain")


            self.dotp_retain = self.compute_total_gradient_dot_product(self.flattened_gradient, self.structure_map, 
                                                                           self.flattened_memory, self.structure_map)
            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss,
            }
            print_metrics(self.steps, metrics)
    
            del retain_loss


        elif self.loss_type == 'npo_grukl':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 
            if self.ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 
            objective_loss = loss
            self.store_grads(model, loss=objective_loss, typ = "objective")

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            del retain_outputs
            self.store_grads(model, loss=retain_loss, typ = "retain")

            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss,
            }
            print_metrics(self.steps, metrics)
  
            del retain_loss

        elif self.loss_type in ['wga',"wga_gru"]:
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weights = (- lm_loss).exp().detach() ** self.beta
            loss = -(weights * lm_loss)[shift_labels.view(-1)!=-100].mean()

            retain_loss = 0
            if self.gradient_edit:
                objective_loss = loss
                self.store_grads(model, loss=objective_loss, typ = "objective")
                retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
                retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
                retain_loss = retain_outputs.loss
                del retain_outputs
                self.store_grads(model, loss=retain_loss, typ = "retain")
                
            self.steps +=1
            metrics = {
                'Loss': loss,
                'retain_loss': retain_loss,
            }
            print_metrics(self.steps, metrics)
    
            del retain_loss


        if self.with_KL_gradient:
            _, retain_inputs = inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)

            self.store_grads(model, loss=retain_loss, typ = "KL_objective")

            print("KL retain loss:",retain_loss)

            del retain_loss
            
      
        return (loss, outputs) if return_outputs else loss

    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        '''
        RZ: Call this function in Trainer.train() when evakluating the performace of each checkpoint.
        '''

        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)

        print('####### Evaluating the model...... #######')
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)

        forget_rate = eval_cfg.split_list[-1].split('_')[0]

        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):
                
                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")

                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)

                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)

                # if int(os.environ.get('RANK', '0')) == 0:
                #    import pdb; pdb.set_trace()

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader)
                
                # kl_divergence_log = get_kl_divergence(model, self.oracle_model, eval_dataloader)
                # eval_logs['kl_divergence'] = kl_divergence_log

                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts

                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))

                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)
                            #delete old files use shutil
                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)

                else:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}.json")))
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs
                                
            if self.accelerator.is_local_main_process:

                aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)
                
                model_utility = get_model_utility(aggregated_eval_logs)
                retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                forget_quality, trust_ratio = get_forget_quality(aggregated_eval_logs, retain_result)
                aaggregate_stat = {**model_utility, **forget_quality}

                aaggregate_stat['curr_step'] = curr_step
                aaggregate_stat['seed'] = self.seed
                aaggregate_stat['loss_type'] = self.loss_type

                with open(os.path.join(curr_save_dir, "aggregate_stat.txt"), 'w') as txtfile:
                    for key, value in aaggregate_stat.items():
                        txtfile.write(f"{key}: {value}\n")

                with open(os.path.join(curr_save_dir, "truth_ratio.pkl"), 'wb') as picklefile:
                    pickle.dump(trust_ratio, picklefile)

class CustomTrainerRetraining(Trainer):
    def __init__(self, *args, **kwargs):
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.seed = kwargs.pop('seed')
        super(CustomTrainerRetraining, self).__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.state.global_step)
        print(f'Generator........Epoch-{self.state.global_step}')

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["generator"] = generator
            dataloader_params["shuffle"] = True # set shuffle=True with specified generator.
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):

        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)

        print('####### Evaluating the model...... #######')
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)

        forget_rate = eval_cfg.split.split('_')[0]

        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):

                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")

                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                # print(save_filename)
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader)
                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts

                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))

                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)
                            #delete old files use shutil
                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)

                else:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}.json")))
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs
                                
            if self.accelerator.is_local_main_process:

                aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)
                
def custom_data_collator_forget(samples):
    rets = []
    if len(samples[0]) == 3:
        idk_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
        data_types = ["idk", "forget", "retain"]
    elif len(samples[0]) == 2:
        forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
        data_types = ["forget", "retain"]
    for data_type in data_types:
        if data_type == "forget":
            data = forget_samples 
        elif data_type == "retain":
            data = retain_samples 
        elif data_type == "idk":
            data = idk_samples
         
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss


def custom_data_collator_forget_wmdp(samples):
    rets = []
    if len(samples[0]) == 3:
        idk_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
        data_types = ["idk", "forget", "retain"]
    elif len(samples[0]) == 2:
        forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
        data_types = ["forget", "retain"]
    for data_type in data_types:
        if data_type == "forget":
            data = forget_samples 
        elif data_type == "retain":
            data = retain_samples 
        elif data_type == "idk":
            data = idk_samples
        
        input_ids = [s["input_ids"] for s in data]
        labels = [s["labels"] for s in data]
        attention_mask = [s["attention_mask"] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets