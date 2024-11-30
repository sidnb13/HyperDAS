import gc
import os
import random
from datetime import datetime

import hydra
import torch
import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from logger import get_logger
from src.hyperdas.data_utils import get_ravel_collate_fn
from src.mdas.llama3.model import RavelMDASNetwork

logger = get_logger(__name__)

load_dotenv(override=True)


def run_experiment(
    config: DictConfig,
    device: str | torch.DeviceObjType = "cuda",
):
    # If experiment config exists, use it instead of root config
    config = config.experiment if hasattr(config, "experiment") else config
    logger.info(f"Config: {OmegaConf.to_yaml(config)}")

    config.wandb_config.run_name = (
        config.wandb_config.run_name
        or f"MDAS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    if config.wandb_config.log:
        wandb.init(
            project=config.wandb_config.project,
            entity=config.wandb_config.entity,
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
        source_suffix_visibility=True,
        base_suffix_visibility=True,
        add_space_before_target=True,
        contain_entity_position=True,
    )

    data_loader = DataLoader(
        train_set,
        batch_size=config.training.train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    test_data_loader = DataLoader(
        test_set,
        batch_size=config.training.test_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    model = RavelMDASNetwork(
        model_name_or_path=config.model.name_or_path,
        intervention_layer=config.model.intervention_layer,
        das_dimension=config.model.das_dimension,
        intervention_location=config.model.intervention_location,
    )

    model = model.to(device)

    model.run_train(
        train_loader=data_loader,
        test_loader=test_data_loader,
        epochs=config.training.n_epochs,
        checkpoint_per_steps=config.training.checkpoint_per_steps,
        eval_per_steps=config.training.eval_per_steps,
        save_dir=config.training.save_dir,
        causal_loss_weight=config.training.causal_loss_weight,
        lr=config.training.lr,
    )

    if config.wandb_config.log:
        wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
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


if __name__ == "__main__":
    main()
