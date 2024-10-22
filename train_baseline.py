import argparse
import gc
import os

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
from src.hyperdas.data_utils import get_ravel_collate_fn
from src.mdas.llama3.model import RavelMDASNetwork

logger = get_logger(__name__)

load_dotenv()


def run_experiment(
    config: DictConfig,
    device: str | torch.DeviceObjType = "cuda",
):
    if config.log_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=OmegaConf.to_container(config),
            group=config.wandb_group,
            tags=config.wandb_tags,
            notes=config.wandb_notes,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_set = load_from_disk(config.train_path)
    test_set = load_from_disk(config.test_path)

    collate_fn = get_ravel_collate_fn(
        tokenizer,
        source_suffix_visibility=True,
        base_suffix_visibility=True,
        add_space_before_target=True,
        contain_entity_position=True,
    )

    data_loader = DataLoader(
        train_set,
        batch_size=config.train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    test_data_loader = DataLoader(
        test_set, batch_size=config.test_batch_size, collate_fn=collate_fn, shuffle=True
    )

    model = RavelMDASNetwork(
        model_name_or_path=config.model_name_or_path,
        intervention_layer=config.intervention_layer,
        das_dimension=config.das_dimension,
        intervention_location=config.intervention_location,
    )

    model = model.to(device)

    model.run_train(
        train_loader=data_loader,
        test_loader=test_data_loader,
        epochs=config.n_epochs,
        checkpoint_per_steps=config.checkpoint_per_steps,
        eval_per_steps=config.eval_per_steps,
        save_dir=config.save_dir,
        causal_loss_weight=config.causal_loss_weight,
        lr=config.lr,
    )

    if config.log_wandb:
        wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="config")
def hydra_main(cfg: DictConfig):
    try:
        logger.info(f"Working directory : {os.getcwd()}")
        logger.info(f"Original working directory    : {get_original_cwd()}")
        logger.info(f"to_absolute_path('foo')       : {to_absolute_path('foo')}")
        logger.info(f"to_absolute_path('/foo')      : {to_absolute_path('/foo')}")
        logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

        # Get the total number of GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs: {num_gpus}")

        # Get the current job number from Hydra's multi-run counter
        try:
            job_num = getattr(HydraConfig.get().job, "num", 0)
            logger.info(f"Job number: {job_num}")
        except Exception:
            # If we're not in a multirun, job.num doesn't exist
            job_num = 0
            logger.debug("Not in a multirun, defaulting job number to 0")

        # Get a unique job identifier (output directory)
        job_id = HydraConfig.get().run.dir
        logger.info(f"Job ID: {job_id}")

        # Assign a GPU based on the job number
        gpu_id = job_num % num_gpus
        device = f"cuda:{gpu_id}"
        logger.info(f"Assigned device: {device}")

        torch.cuda.set_device(device)

        # Pass the device to your run_experiment function
        logger.info(f"Launching job {job_id} on GPU {device}")
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

    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument(
        "--model_name_or_path", type=str, default="/nlp/scr/sjd24/llama3-8b"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--test_path",
        type=str,
        default="./experiments/RAVEL/data/nobel_prize_winner_field_test",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="./experiments/RAVEL/data/nobel_prize_winner_field_train",
    )
    parser.add_argument("--causal_loss_weight", type=float, default=1)

    parser.add_argument(
        "--intervention_location",
        type=str,
        choices=["last_token", "last_entity_token"],
        default="last_entity_token",
    )

    # if None, use Boundless DAS
    parser.add_argument("--das_dimension", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_per_steps", type=int, default=100)
    parser.add_argument("--checkpoint_per_steps", type=int, default=None)

    args = parser.parse_args()
    args = dict(args.__dict__)
    run_experiment(**args)


if __name__ == "__main__":
    use_hydra = os.environ.get("USE_HYDRA", "true").lower() == "true"

    if use_hydra:
        # Use Hydra
        hydra_main()
    else:
        # Use argparse
        argparse_main()
