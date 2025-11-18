import os
import random
import logging
from pathlib import Path

from sympy import false
import torch
import pytorch_lightning as L
from omegaconf import OmegaConf, open_dict
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from utils import seed
from models.ebmpc import EBMPC

log = logging.getLogger(__name__)

ALL_MODEL_KEYS = ["encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"]

def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, cfg, device):
    """Load the world model components from checkpoint or instantiate."""
    result = {}
    if model_ckpt.exists():
        ckpt = load_ckpt(model_ckpt, device)
        result.update(ckpt)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(cfg.encoder)
    if "predictor" not in result:
        result["predictor"] = hydra.utils.instantiate(cfg.predictor)
    if cfg.has_decoder and "decoder" not in result:
        if cfg.env.decoder_path is not None:
            decoder_path = os.path.join(os.getcwd(), cfg.env.decoder_path)
            ckpt = torch.load(decoder_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = ckpt
        else:
            raise ValueError("Decoder path not provided but required")
    else:
        result["decoder"] = None

    for k in ["encoder", "predictor", "decoder"]:
        if result[k] is not None:
            for p in result[k].parameters():
                p.requires_grad = False

    return result


@hydra.main(config_path="conf", config_name="train_ebmpc")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg.saved_folder = os.getcwd()
    seed(cfg.training.seed)
    log.info(f"Training directory: {cfg.saved_folder}")

    model_path = Path(cfg.world_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"World model path not found: {model_path}")

    wm_cfg_path = model_path / "hydra.yaml"
    if not wm_cfg_path.exists():
        raise FileNotFoundError(f"World model config not found: {wm_cfg_path}")

    wm_cfg = OmegaConf.load(wm_cfg_path)
    model_ckpt = model_path / "checkpoints" / "model_latest.pth"
    if not model_ckpt.exists():
        raise FileNotFoundError(f"World model checkpoint not found: {model_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_components = load_model(model_ckpt, wm_cfg, device)

    datasets, _ = hydra.utils.call(
        wm_cfg.env.dataset,
        num_hist=wm_cfg.num_hist,
        num_pred=wm_cfg.num_pred,
        frameskip=wm_cfg.frameskip,
    )

    train_loader = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets["valid"],
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    world_model = hydra.utils.instantiate(
        wm_cfg.model,
        encoder=wm_components["encoder"],
        predictor=wm_components["predictor"],
        decoder=wm_components.get("decoder"),
        proprio_encoder=wm_components.get("proprio_encoder"),
        action_encoder=wm_components.get("action_encoder"),
        proprio_dim=wm_cfg.proprio_emb_dim,
        action_dim=wm_cfg.action_emb_dim,
        concat_dim=wm_cfg.concat_dim,
        num_action_repeat=wm_cfg.num_action_repeat,
        num_proprio_repeat=wm_cfg.num_proprio_repeat,
    )
    world_model.eval()


    image_dims = [wm_cfg.img_size, wm_cfg.img_size]
    action_dim = datasets["train"].action_dim

    ebmpc_model = EBMPC(
        world_model=world_model,
        num_mcmc_steps=cfg.model.num_mcmc_steps,
        action_dim=action_dim,
        mcmc_step_size=cfg.model.mcmc_step_size,
        mcmc_step_size_learnable=cfg.model.mcmc_step_size_learnable,
        image_dims=image_dims,
        embedding_dim=cfg.model.embedding_dim,
        num_transformer_blocks=cfg.model.num_transformer_blocks,
        multiheaded_attention_heads=cfg.model.multiheaded_attention_heads,
        ffn_dim_multiplier=cfg.model.ffn_dim_multiplier,
        lr=cfg.training.lr,
        learning=True,
    )

    csv_logger = CSVLogger(save_dir=os.getcwd(), name="logs", version="")
    loggers = [csv_logger]

    if cfg.training.use_wandb:
        wandb_logger = WandbLogger(
            project="train_ebmpc",
            name=f"ebmpc_{wm_cfg.env.name}",
            save_dir=os.getcwd(),
            log_model=False,
        )
        loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="ebmpc-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=loggers,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        accumulate_grad_batches=4
    )

    trainer.fit(ebmpc_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
