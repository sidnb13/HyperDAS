import argparse
import gc
import os
import random

import hydra
import torch
from datasets import load_from_disk
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb
from logger import get_logger
from src.hyperdas.data_utils import (
    get_ravel_collate_fn,
)

load_dotenv()

logger = get_logger(__name__)


def run_experiment(
    config: DictConfig,
    device: str | torch.DeviceObjType = "cuda",
):
    # If experiment config exists, use it instead of root config
    config = config.experiment if hasattr(config, "experiment") else config
    logger.info(f"Config: {OmegaConf.to_yaml(config)}")

    if config.training.debug_model:
        config.model.inference_modes = ["groundtruth"]

    if "default" in config.model.inference_modes:
        config.model.inference_modes.remove("default")
        config.model.inference_modes.append(None)

    if config.wandb_config.log:
        wandb.init(
            project=config.wandb_config.project,
            name=config.wandb_config.run_name,
            config=OmegaConf.to_container(config),
            group=config.wandb_config.group,
            tags=config.wandb_config.tags,
            notes=config.wandb_config.notes,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_set = load_from_disk(config.dataset.train_path)
    test_set = load_from_disk(config.dataset.test_path)

    collate_fn = get_ravel_collate_fn(
        tokenizer,
        source_suffix_visibility=config.dataset.source_suffix_visibility,
        base_suffix_visibility=config.dataset.base_suffix_visibility,
        add_space_before_target=True,
        contain_entity_position="groundtruth" in config.model.inference_modes,
    )

    # very hacky
    try:
        getattr(HydraConfig.get().job, "num")
        is_multirun = True
    except Exception:
        is_multirun = False
    num_workers = 0 if is_multirun else config.training.num_workers

    train_data_loader = DataLoader(
        train_set,
        batch_size=config.training.train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_data_loader = DataLoader(
        test_set,
        batch_size=config.training.test_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    from src.hyperdas.llama3.model import RavelInterpretorHypernetwork

    hypernetwork = RavelInterpretorHypernetwork(config, device)

    if config.training.load_trained_from is not None:
        logger.info(f"Loading model from {config.training.load_trained_from}")
        hypernetwork.load_model(config.training.load_trained_from)

    hypernetwork.run_train(
        train_loader=train_data_loader,
        test_loader=test_data_loader,
        inference_modes=config.model.inference_modes,
        epochs=config.training.n_epochs,
        steps=config.training.n_steps,
        eval_per_steps=config.training.eval_per_steps,
        checkpoint_per_steps=config.training.checkpoint_per_steps,
        save_dir=config.training.save_dir,
        save_model=config.training.save_model,
        debug_model=config.training.debug_model,
        run_name=config.wandb_config.run_name,
    )

    if config.wandb_config.log:
        wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="config")
def hydra_main(cfg: DictConfig):
    try:
        logger.info(f"Working directory : {os.getcwd()}")
        logger.info(f"Original working directory    : {get_original_cwd()}")
        logger.info(f"to_absolute_path('foo')       : {to_absolute_path('foo')}")
        logger.info(f"to_absolute_path('/foo')      : {to_absolute_path('/foo')}")

        # Check if we're running in serial mode
        is_serial = os.environ.get("LAUNCH_MODE", "parallel") == "serial"

        if is_serial:
            # Use single GPU for serial mode
            device = "cuda:0"
        else:
            # Use distributed GPUs for parallel mode
            num_gpus = torch.cuda.device_count()
            try:
                job_num = getattr(HydraConfig.get().job, "num", 0)
            except Exception:
                job_num = 0
            gpu_id = job_num % num_gpus
            device = f"cuda:{gpu_id}"

        logger.info(
            f"Running in {'serial' if is_serial else 'parallel'} mode on device {device}"
        )

        if not is_serial:
            torch.cuda.set_device(device)

        # Set seed
        torch.manual_seed(cfg.training.seed)
        random.seed(cfg.training.seed)

        run_experiment(cfg, device)

        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        logger.error(f"An error occurred in hydra_main: {str(e)}", exc_info=True)
        raise


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
    parser.add_argument("--causal_loss_weight", type=float, default=1)
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
