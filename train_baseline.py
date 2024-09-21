import torch
from torch import compile
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import os
import time
import sys
import wandb
import random
import numpy as np
import json
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from src.data_utils import get_ravel_collate_fn, generate_ravel_dataset_from_filtered
from src.utils import add_fwd_hooks
import argparse
from pyvene import IntervenableConfig, RepresentationConfig, LowRankRotatedSpaceIntervention, IntervenableModel, count_parameters



from torch import optim
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import get_scheduler


def run_experiment(
    log_wandb=True,
    wandb_project="hypernetworks-interpretor",
    wandb_run_name=None,
    intervention_layer=15,
    model_name_or_path="./models/llama3-8b",
    load_trained_from=None,
    batch_size=8,
    save_dir=None,
    n_epochs=1,
    das_dimension=None,
    lr=3e-5,
    eval_per_steps=100,
    checkpoint_per_steps=500,
    test_path=None,
    train_path=None,
    causal_loss_weight=1,
    intervention_location="last_entity_token",
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if log_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "targetmodel": model_name_or_path, 
                "dataset": "ravel",
                "intervention_layer": intervention_layer,
                "das_dimension": das_dimension,
            },
        )
        
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_set = load_from_disk(train_path)
    test_set = load_from_disk(test_path)
                
    collate_fn = get_ravel_collate_fn(
        tokenizer, 
        source_suffix_visibility=True, 
        base_suffix_visibility=True, 
        add_space_before_target=True,
        contain_entity_position=True,
    )
    
    data_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    model = model.to("cuda")
    
    intervention_config = IntervenableConfig(
        model_type=type(model),
        representations=[
        RepresentationConfig(
              intervention_layer,  # layer
              'block_output',  # intervention repr
              "pos",  # intervention unit
              1,  # max number of unit
              das_dimension)
      ],
      intervention_types=LowRankRotatedSpaceIntervention,
    )
    
    intervenable = IntervenableModel(intervention_config, model)
    intervenable.set_device(model.device)
    intervenable.disable_model_gradients()
    
    
    inv_keys = list(intervenable.interventions.keys())[0]
    
    optimizer_params = []
    for k, v in intervenable.interventions.items():
        optimizer_params += [{'params': v[0].rotate_layer.parameters()}]
            
    optimizer = torch.optim.AdamW(optimizer_params,
                                    lr=lr,
                                    weight_decay=0)

    total_steps = len(data_loader) * n_epochs
    scheduler = get_scheduler(
        'constant',
        optimizer=optimizer,
        num_training_steps=total_steps
    )
    
    print("Model trainable parameters: ", count_parameters(intervenable.model))
    print("Intervention trainable parameters: ", intervenable.count_parameters())

    cur_steps = 0
    
    def forward(
        base_input_ids: torch.Tensor = None,
        base_attention_mask: torch.Tensor = None,
        base_intervention_position: torch.Tensor = None,
        base_position_ids: torch.Tensor = None,
        source_input_ids: torch.Tensor = None,
        source_attention_mask: torch.Tensor = None,
        source_intervention_position: torch.Tensor = None,
        source_position_ids: torch.Tensor = None,
        intervention_layer: int = None,
    ):
        if intervention_layer is None:
            raise ValueError("intervention_layer must be specified")
        
        if base_position_ids is None:
            # 0 for all the padding tokens and start from 1 for the rest
            base_position_ids = torch.cumsum(base_attention_mask, dim=1) * base_attention_mask
        
        if source_position_ids is None:
            source_position_ids = torch.cumsum(source_attention_mask, dim=1) * source_attention_mask
        
        # print(source_intervention_position.unsqueeze(0).shape, base_intervention_position.unsqueeze(0).shape)
        b_s = base_input_ids.shape[0]
        intervention_locations = {
            "sources->base": (
                source_intervention_position.unsqueeze(0).unsqueeze(-1),
                base_intervention_position.unsqueeze(0).unsqueeze(-1)
            )
        }
        
        _, counterfactual_outputs = intervenable(
            {
                "input_ids": base_input_ids,
                'attention_mask': base_attention_mask,
                'position_ids': base_position_ids
            }, [
                {
                    "input_ids": source_input_ids,
                    'attention_mask': source_attention_mask,
                    'position_ids': source_position_ids
                }
            ] , intervention_locations
        )
        
        return counterfactual_outputs
        
                
    def eval_accuracy(test_loader, eval_n_label_tokens=3):
        
        intervenable.eval()
        correct_idxs = []
        is_causal = []
        
        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                
                if intervention_location == "last_entity_token":
                    base_intervention_position = batch["base_entity_position_ids"].to("cuda") 
                    source_intervention_position = batch["source_entity_position_ids"].to("cuda")
                else:
                    base_intervention_position = batch["base_input_ids"].shape[1] - 1
                    source_intervention_position = batch["source_input_ids"].shape[1] - 1
                    
                    base_intervention_position = torch.tensor([base_intervention_position] * batch["base_input_ids"].shape[0]).to("cuda")
                    source_intervention_position = torch.tensor([source_intervention_position] * batch["source_input_ids"].shape[0]).to("cuda")
                
                output = forward(
                    base_input_ids=batch["base_input_ids"].to("cuda"),
                    base_attention_mask=batch["base_attention_mask"].to("cuda"),
                    base_intervention_position=base_intervention_position,
                    source_input_ids=batch["source_input_ids"].to("cuda"),
                    source_attention_mask=batch["source_attention_mask"].to("cuda"),
                    source_intervention_position=source_intervention_position,
                    intervention_layer=intervention_layer,
                )
                
                logits = output.logits
                                
                batch_pred_ids = torch.argmax(logits, dim=-1)
                is_causal.extend(batch["is_causal"].cpu().numpy().tolist())
                
                for i, (label, pred_ids) in enumerate(zip(batch["labels"].to("cuda"), batch_pred_ids)):
                    label_idx = label != -100
                    output_idx = torch.zeros_like(label_idx)
                    output_idx[:-1] = label_idx[1:]
                    
                    label = label[label_idx]
                    pred_ids = pred_ids[output_idx]
                    
                    if eval_n_label_tokens is not None and len(label) > eval_n_label_tokens:
                        label = label[:eval_n_label_tokens]
                        pred_ids = pred_ids[:eval_n_label_tokens]
                    
                    is_correct = (torch.sum (label == pred_ids) == torch.numel(label)).item()    
                    if is_correct:
                        correct_idxs.append(batch_id * len(batch["labels"]) + i)
                
                
        total_causal = sum(is_causal)
        total_isolate = len(is_causal) - total_causal
        
        correct_causal = sum([is_causal[i] for i in correct_idxs])
        correct_isolate = len(correct_idxs) - correct_causal
        
        causal_acc = correct_causal / total_causal if total_causal > 0 else 0.0
        isolate_acc = correct_isolate / total_isolate if total_isolate > 0 else 0.0
        
        disentangle_acc = 0.5 * (causal_acc + isolate_acc) if total_isolate > 0 else causal_acc
        
        accuracies = {
            "causal": causal_acc,
            "isolate": isolate_acc,
            "disentangle": disentangle_acc
        }
                    
        return accuracies
    
    for epoch in range(n_epochs):
        # Create a tqdm progress bar
        with tqdm(
            total=len(data_loader),
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            unit="batch",
            disable=True,
        ) as pbar:
            num_datapoints_in_epoch = 0
            epoch_train_loss = 0
            epoch_gradient_norm = 0
            # Train loop
            for step, batch in enumerate(
                data_loader
            ):  
                if eval_per_steps is not None:
                    if cur_steps % eval_per_steps == 0:
                        accuracies = eval_accuracy(
                            test_data_loader, eval_n_label_tokens=3
                        )
                        
                        causal_acc = accuracies["causal"]
                        isolate_acc = accuracies["isolate"]
                        disentangle_acc = accuracies["disentangle"]
                                                
                        if wandb.run:
                            wandb.log(
                                {
                                    "causal_accuracy": causal_acc,
                                    "isolate_accuracy": isolate_acc,
                                    "disentangle_accuracy": disentangle_acc,
                                }
                            )
                        
                        print(f"Disentangle Acc: {disentangle_acc}, Causal Acc: {causal_acc}, Isolate Acc: {isolate_acc}")
                    
                if checkpoint_per_steps is not None:
                    if cur_steps % checkpoint_per_steps == 0 and save_dir is not None:
                        print("Saving model to {}".format(os.path.join(save_dir, f"model_epoch_{epoch}_step_{step}")))
                        torch.save(intervenable.interventions[inv_keys][0].state_dict(), os.path.join(save_dir, "das_epoch_{epoch}_step_{step}.pt"))
                
                current_batch_size = len(batch["labels"])
                num_datapoints_in_epoch += current_batch_size
                
                training_loss = 0
                
                if intervention_location == "last_entity_token":
                    base_intervention_position = batch["base_entity_position_ids"].to("cuda") 
                    source_intervention_position = batch["source_entity_position_ids"].to("cuda")
                else:
                    base_intervention_position = batch["base_input_ids"].shape[1] - 1
                    source_intervention_position = batch["source_input_ids"].shape[1] - 1
                    
                    base_intervention_position = torch.tensor([base_intervention_position] * batch["base_input_ids"].shape[0]).to("cuda")
                    source_intervention_position = torch.tensor([source_intervention_position] * batch["source_input_ids"].shape[0]).to("cuda")
                
                output = forward(
                    base_input_ids=batch["base_input_ids"].to("cuda"),
                    base_attention_mask=batch["base_attention_mask"].to("cuda"),
                    base_intervention_position=base_intervention_position,
                    source_input_ids=batch["source_input_ids"].to("cuda"),
                    source_attention_mask=batch["source_attention_mask"].to("cuda"),
                    source_intervention_position=source_intervention_position,
                    intervention_layer=intervention_layer,
                )
                                
                logits = output.logits
                labels = batch["labels"].to("cuda")
                
                log_prob_predictions = torch.nn.functional.log_softmax(
                    logits.reshape(-1, logits.shape[-1]),
                    dim=1,
                )
                
                loss_weight = torch.ones_like(labels, dtype=log_prob_predictions.dtype)
                loss_weight[batch["is_causal"].to("cuda"), :] = causal_loss_weight
                
                labels = labels.reshape(-1)
                
                loss_weight = loss_weight.reshape(-1)

                assert labels.shape == log_prob_predictions.shape[:-1]
                
                # Only consider the tokens that are not -100 in target_labels
                label_indices = labels != -100
                output_idices = torch.zeros_like(label_indices)
                output_idices[:-1] = label_indices[1:]
                
                log_prob_predictions = log_prob_predictions[output_idices, :]
            
                labels = labels[label_indices]
                
                # Compute the cross-entropy loss with masking
                
                loss_weight = loss_weight[label_indices]
                criterion = torch.nn.CrossEntropyLoss(reduction="none")
                loss = criterion(log_prob_predictions, labels.long())
                loss = (loss * loss_weight).mean()

                prediction_loss = loss                          
                training_loss += prediction_loss
                                    
                training_loss.backward()
                            
                # metrics
                epoch_train_loss += training_loss.item() * current_batch_size
                
                optimizer.step()
                optimizer.zero_grad()
                
                # TEST: orthogonalize the rotation matrix every step
                """if self.use_das_intervention:
                    self.interpretor.das_module.orthogonalize_rotation_matrix()"""

                metrics = {
                    "step": cur_steps,
                    "train_batch_total_loss": training_loss.item(),
                    "train_batch_prediction_loss": prediction_loss.item(),
                }

                if wandb.run:
                    wandb.log(metrics)
                if cur_steps % 5 == 0:
                    print(metrics)

                # Update progress bar
                pbar.update(1)  # note: this was incorrectly displaying before!
                cur_steps += 1
            
            if wandb.run:
                wandb.log(
                    {
                        "epoch_train_total_loss": epoch_train_loss
                        / num_datapoints_in_epoch,
                    }
                )
    
    accuracies = eval_accuracy(
        test_data_loader, eval_n_label_tokens=3
    )
    for k, v in accuracies.items():
        print(f"{k}: {v}")
    
    if save_dir is not None:
        torch.save(intervenable.interventions[inv_keys][0].state_dict(), os.path.join(save_dir, "final_das_module.pt"))
        json.dump(accuracies, open(os.path.join(save_dir, "final_accuracies.json"), "w"))
        
        

    if log_wandb:
        wandb.finish()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="HyperDAS")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--intervention_layer", type=int, default=15)
    
    parser.add_argument("--load_trained_from", type=str, default=None)
    
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--model_name_or_path", type=str, default="/nlp/scr/sjd24/llama3-8b")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--test_path", type=str, default="./experiments/ravel/data/nobel_prize_winner_field_test")
    parser.add_argument("--train_path", type=str, default="./experiments/ravel/data/nobel_prize_winner_field_train")
    parser.add_argument("--causal_loss_weight", type=float, default=1)
    
    parser.add_argument("--intervention_location", type=str, choices=["last_token", "last_entity_token"], default="last_entity_token")
        
    # if None, use Boundless DAS
    parser.add_argument("--das_dimension", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_per_steps", type=int, default=100)
    parser.add_argument("--checkpoint_per_steps", type=int, default=None)
    
    args = parser.parse_args()
    args = dict(args.__dict__)
    run_experiment(**args)