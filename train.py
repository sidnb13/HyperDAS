import argparse
import os

import hydra
import torch
from datasets import load_from_disk
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb
from logger import get_logger
from src.hyperdas.data_utils import (
    get_ravel_collate_fn,
)

logger = get_logger(__name__)


def run_experiment(
    config: DictConfig,
    device: str | torch.DeviceObjType = "cuda",
):
    """if save_dir is not None:
    save_dir = os.path.join("./models", save_dir)"""
    if config.log_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=OmegaConf.to_container(config),
            group=config.wandb_group,
            tags=config.wandb_tags,
            notes=config.wandb_notes,
        )

    if "default" in config.inference_modes:
        config.inference_modes.remove("default")
        config.inference_modes.append(None)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_set = load_from_disk(config.train_path)
    test_set = load_from_disk(config.test_path)

    # train_set = Dataset.from_list([d for d in train_set if d["attribute_type"] == "causal"])
    # test_set = Dataset.from_list([d for d in test_set if d["attribute_type"] == "causal"])

    collate_fn = get_ravel_collate_fn(
        tokenizer,
        source_suffix_visibility=config.source_suffix_visibility,
        base_suffix_visibility=config.base_suffix_visibility,
        add_space_before_target=True,
        contain_entity_position="groundtruth" in config.inference_modes,
    )

    data_loader = DataLoader(
        train_set, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True
    )

    test_data_loader = DataLoader(
        test_set, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True
    )

    from src.hyperdas.llama3.model import RavelInterpretorHypernetwork

    hypernetwork = RavelInterpretorHypernetwork(
        model_name_or_path=config.model_name_or_path,
        num_editing_heads=32,
        intervention_layer=config.intervention_layer,
        subspace_module=config.subspace_module,
        das_dimension=config.das_dimension,
        chop_editor_at_layer=config.num_decoders,
        initialize_from_scratch=config.initialize_from_scratch,
        ablate_base_token_attention=config.ablate_base_token_attention,
        ablate_source_token_attention=config.ablate_source_token_attention,
        break_asymmetric=config.break_asymmetric,
        top_k_parameter=config.top_k_parameter,
        lambda_parameter=config.lambda_parameter,
        epsilon=config.epsilon,
        importance_power=config.importance_power,
        device=device,
        compute_metrics=config.compute_metrics,
    )

    if config.load_trained_from is not None:
        hypernetwork.load_model(config.load_trained_from)

    # current problem: 1728 / 30864
    hypernetwork.run_train(
        train_loader=data_loader,
        test_loader=test_data_loader,
        inference_modes=config.inference_modes,
        epochs=config.n_epochs,
        steps=config.n_steps,
        checkpoint_per_steps=config.checkpoint_per_steps,
        eval_per_steps=config.eval_per_steps,
        save_dir=config.save_dir,
        apply_source_selection_sparsity_loss=config.source_selection_sparsity_loss,
        sparsity_loss_weight_start=config.sparsity_loss_weight_start,
        sparsity_loss_weight_end=config.sparsity_loss_weight_end,
        sparsity_loss_warm_up_ratio=config.sparsity_loss_warm_up_ratio,
        causal_loss_weight=config.causal_loss_weight,
        iso_loss_weight=config.iso_loss_weight,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr=config.lr,
        save_model=config.save_model,
        target_intervention_num=config.target_intervention_num,
    )

    if config.log_wandb:
        wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="config")
def hydra_main(cfg: DictConfig):
    # Get the total number of GPUs
    num_gpus = torch.cuda.device_count()
    # Get the current job number from Hydra's multi-run counter
    job_num = HydraConfig.get().job.num
    # Get a unique job identifier (output directory)
    job_id = HydraConfig.get().run.dir
    # Assign a GPU based on the job number
    gpu_id = job_num % num_gpus
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Pass the device to your run_experiment function
    logger.debug("Launching job %s on GPU %s", job_id, device)
    run_experiment(cfg, device)


def argparse_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="HyperDAS")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--intervention_layer", type=int, default=15)

    parser.add_argument("--load_trained_from", type=str, default=None)

    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--n_steps", type=int, default=-1)
    parser.add_argument(
        "--model_name_or_path", type=str, default="/scr-ssd/sjd24/llama3-8b"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--source_suffix_visibility", default=False, action="store_true"
    )
    parser.add_argument("--base_suffix_visibility", default=False, action="store_true")

    parser.add_argument(
        "--test_path", type=str, default="./experiments/RAVEL/data/city_test_small"
    )
    parser.add_argument(
        "--train_path", type=str, default="./experiments/RAVEL/data/city_train"
    )

    parser.add_argument("--source_selection_sparsity_loss", type=bool, default=True)
    parser.add_argument("--sparsity_loss_warm_up_ratio", type=float, default=0.25)
    parser.add_argument("--sparsity_loss_weight_start", type=float, default=0.0)
    parser.add_argument("--sparsity_loss_weight_end", type=float, default=1.0)

    parser.add_argument("--target_intervention_num", type=int, default=None)
    parser.add_argument("--causal_loss_weight", type=float, default=3.5)
    parser.add_argument("--iso_loss_weight", type=float, default=1)

    parser.add_argument(
        "--save_dir", type=str, default="/scr-ssd/sjd24/city_entropy_1014"
    )
    parser.add_argument("--save_model", default=False, action="store_true")

    parser.add_argument(
        "--inference_modes", nargs="+", default=["default", "bidding_argmax"]
    )

    # Ablation
    parser.add_argument("--num_decoders", type=int, default=8)
    parser.add_argument("--initialize_from_scratch", default=False, action="store_true")
    parser.add_argument(
        "--ablate_base_token_attention", default=False, action="store_true"
    )
    parser.add_argument(
        "--ablate_source_token_attention", default=False, action="store_true"
    )
    parser.add_argument("--break_asymmetric", default=False, action="store_true")

    # if None, use Boundless DAS
    parser.add_argument(
        "--subspace_module",
        default="ReflectSelect",
        choices=[
            None,
            "DAS",
            "BoundlessDAS",
            "MaskSelect",
            "ReflectSelect",
            "QuasiProjective",
        ],
    )
    parser.add_argument("--das_dimension", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_per_steps", type=int, default=2000)
    parser.add_argument("--checkpoint_per_steps", type=int, default=2000)

    args = parser.parse_args()
    args = dict(args.__dict__)
    run_experiment(OmegaConf.create(args))


if __name__ == "__main__":
    use_hydra = os.environ.get("USE_HYDRA", "true").lower() == "true"

    if use_hydra:
        # Use Hydra
        hydra_main()
    else:
        # Use argparse
        argparse_main()
