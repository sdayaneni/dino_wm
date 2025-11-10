import os
import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta, datetime
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# Setup logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Trainer:
    def __init__(self, cfg):
        print("=" * 80)
        print("INITIALIZING TRAINER")
        print("=" * 80)
        
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        # Log initialization info
        print(f"Model saved directory: {cfg['saved_folder']}")
        print(f"Model name: {model_name}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Debug mode handling
        if getattr(self.cfg, "debug", False):
            print("=" * 80)
            print("WARNING: Running in DEBUG MODE")
            print("=" * 80)

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            print("=" * 80)
            print("MULTIRUN SETUP (SLURM)")
            print("=" * 80)
            print(f"SLURM_JOB_NODELIST: {os.environ.get('SLURM_JOB_NODELIST', 'N/A')}")
            print(f"SLURM_JOBID: {os.environ.get('SLURM_JOBID', 'N/A')}")
            print(f"SLURM_NTASKS: {os.environ.get('SLURM_NTASKS', 'N/A')}")
            print(f"SLURM_GPUS_PER_NODE: {os.environ.get('SLURM_GPUS_PER_NODE', 'N/A')}")
            print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
            if 'DEBUGVAR' in os.environ:
                print(f"DEBUGVAR: {os.environ['DEBUGVAR']}")
            
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                print("Initializing DDP process group...")
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                print(" DDP setup completed successfully")
            except Exception as e:
                print(f" DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            print("=" * 80)
            # # ==== /init ddp process group ====

        print("=" * 80)
        print("INITIALIZING ACCELERATOR")
        print("=" * 80)
        self.accelerator = Accelerator(log_with="wandb")
        print(f"Process rank: {self.accelerator.local_process_index}")
        print(f"Number of processes: {self.accelerator.num_processes}")
        print(f"Is main process: {self.accelerator.is_main_process}")
        self.device = self.accelerator.device
        print(f"Device: {self.device}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0
        self.global_step = 0
        self.log_every_steps = getattr(self.cfg.training, "log_every_steps", 0)

        # Batch size validation
        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)
        
        print(f"Total batch size: {cfg.effective_batch_size}")
        print(f"Batch size per GPU: {cfg.gpu_batch_size}")
        print(f"Number of GPUs: {self.accelerator.num_processes}")
        print(f"Effective batch size: {cfg.effective_batch_size}")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print("=" * 80)
            print("INITIALIZING WANDB")
            print("=" * 80)
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg.get("wandb_run_id", None)
                if wandb_run_id:
                    print(f"Resuming WandB run: {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if getattr(self.cfg, "debug", False):
                print("Initializing WandB in DEBUG project...")
                self.wandb_run = wandb.init(
                    project="dino_wm_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                print("Initializing WandB...")
                self.wandb_run = wandb.init(
                    project="dino_wm",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            print(f"WandB run name: {model_name}")
            print(f"WandB run ID: {self.wandb_run.id}")
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))
            print(" WandB initialized successfully")
            
            # Log initial configuration and hardware info to WandB
            self.wandb_run.log({
                "config/num_hist": self.cfg.num_hist,
                "config/num_pred": self.cfg.num_pred,
                "config/frameskip": self.cfg.frameskip,
                "config/env_name": self.cfg.env.name,
                "config/training_epochs": self.cfg.training.epochs,
                "config/training_batch_size": self.cfg.training.batch_size,
                "config/encoder_lr": self.cfg.training.encoder_lr,
                "config/predictor_lr": getattr(self.cfg.training, "predictor_lr", 0),
                "config/decoder_lr": getattr(self.cfg.training, "decoder_lr", 0),
                "config/action_encoder_lr": getattr(self.cfg.training, "action_encoder_lr", 0),
                "hardware/num_processes": self.accelerator.num_processes,
                "hardware/device": str(self.device),
                "hardware/cuda_available": torch.cuda.is_available(),
                "hardware/cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "hardware/effective_batch_size": self.cfg.effective_batch_size,
                "hardware/batch_size_per_gpu": self.cfg.gpu_batch_size,
            })

        # Set random seed
        seed(cfg.training.seed)
        print(f"Random seed set to: {cfg.training.seed}")
        
        print("=" * 80)
        print("LOADING DATASETS")
        print("=" * 80)
        print(f"Dataset path: {self.cfg.env.dataset.data_path}")
        print(f"Number of history frames: {self.cfg.num_hist}")
        print(f"Number of prediction frames: {self.cfg.num_pred}")
        print(f"Frameskip: {self.cfg.frameskip}")
        
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]
        
        print(f"Train dataset size: {len(self.datasets['train'])}")
        print(f"Validation dataset size: {len(self.datasets['valid'])}")
        print(f"Train trajectory dataset size: {len(self.train_traj_dset)}")
        print(f"Validation trajectory dataset size: {len(self.val_traj_dset)}")
        
        # Log dataset info to WandB
        if self.accelerator.is_main_process:
            self.wandb_run.log({
                "dataset/train_size": len(self.datasets['train']),
                "dataset/val_size": len(self.datasets['valid']),
                "dataset/train_traj_size": len(self.train_traj_dset),
                "dataset/val_traj_size": len(self.val_traj_dset),
            })

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False, # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
            )
            for x in ["train", "valid"]
        }

        print(f"Dataloader batch size per GPU: {self.cfg.gpu_batch_size}")
        print(f"Number of workers per GPU: {self.cfg.env.num_workers}")
        print(f"Total workers: {self.cfg.env.num_workers * self.accelerator.num_processes}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )
        print(" Dataloaders prepared")

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        
        print("=" * 80)
        print("MODEL TRAINING FLAGS")
        print("=" * 80)
        print(f"Train encoder: {self.train_encoder}")
        print(f"Train predictor: {self.train_predictor}")
        print(f"Train decoder: {self.train_decoder}")
        print(f"Has predictor: {self.cfg.has_predictor}")
        print(f"Has decoder: {self.cfg.has_decoder}")

        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            ["decoder", "decoder_optimizer"] if self.train_decoder else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        print("=" * 80)
        print("INITIALIZING MODELS")
        print("=" * 80)
        self.init_models()
        
        print("=" * 80)
        print("INITIALIZING OPTIMIZERS")
        print("=" * 80)
        self.init_optimizers()
        
        # Log non-trainable parameters
        print("=" * 80)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 80)
        total_params = 0
        trainable_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                print(f"Non-trainable: {name} (shape: {param.shape})")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Log model info to WandB
        if self.accelerator.is_main_process:
            self.wandb_run.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/non_trainable_parameters": total_params - trainable_params,
                "model/train_encoder": self.train_encoder,
                "model/train_predictor": self.train_predictor,
                "model/train_decoder": self.train_decoder,
                "model/has_predictor": self.cfg.has_predictor,
                "model/has_decoder": self.cfg.has_decoder,
            })
        
        self.epoch_log = OrderedDict()
        
        
        # Gradient Accumulation
        # Useful for simulating larger batch sizes when memory is limited
        # Uncomment the following lines to enable:
        # self.accumulate_grad_batches = getattr(self.cfg.training, "accumulate_grad_batches", 1)
        # self.accumulation_step = 0
        # if self.accumulate_grad_batches > 1:
        #     print(f"Gradient accumulation enabled: {self.accumulate_grad_batches} batches")
        
        # Gradient Clipping
        # Helps with training stability by preventing exploding gradients
        # Uncomment the following lines to enable:
        # self.gradient_clip_val = getattr(self.cfg.training, "gradient_clip_val", 0.0)
        # if self.gradient_clip_val > 0:
        #     print(f"Gradient clipping enabled: max_norm={self.gradient_clip_val}")
        
        # Learning Rate Scheduling
        # Automatically adjusts learning rate during training
        # Uncomment the following lines in init_optimizers() after creating optimizers:
        # from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
        # self.schedulers = {}
        # if self.train_encoder:
        #     self.schedulers['encoder'] = CosineAnnealingLR(
        #         self.encoder_optimizer,
        #         T_max=self.total_epochs,
        #         eta_min=getattr(self.cfg.training, "min_lr", 0)
        #     )
        # # Add similar for other optimizers...
        # # Then in run(), after each epoch:
        # # for name, scheduler in self.schedulers.items():
        # #     scheduler.step()
        # #     if self.accelerator.is_main_process:
        # #         self.wandb_run.log({f"lr/{name}": scheduler.get_last_lr()[0]})
        
        # Early Stopping
        # Stops training if validation loss doesn't improve for N epochs
        # Uncomment the following lines to enable:
        # self.early_stop_patience = getattr(self.cfg.training, "early_stop_patience", 0)
        # self.best_val_loss = float('inf')
        # self.patience_counter = 0
        # if self.early_stop_patience > 0:
        #     print(f"Early stopping enabled: patience={self.early_stop_patience} epochs")
        
        # Mixed Precision Training
        # self.use_amp = getattr(self.cfg.training, "use_amp", False)
        # if self.use_amp:
        #     self.scaler = torch.cuda.amp.GradScaler()
        #     print("Mixed precision training (AMP) enabled")
        
        # Best Model Tracking
        self.best_val_loss = float('inf')
        self.keep_top_k_checkpoints = getattr(self.cfg.training, "keep_top_k_checkpoints", 5)
        self.checkpoint_history = []
        print(f"Best model tracking enabled (keeping top {self.keep_top_k_checkpoints} checkpoints)")
        
        # Memory Management
        self.memory_cleanup_interval = getattr(self.cfg.training, "memory_cleanup_interval", 100)
        if self.memory_cleanup_interval > 0:
            print(f"Memory cleanup enabled: every {self.memory_cleanup_interval} batches")
        
        # Configurable Validation Frequency
        self.val_every_n_epochs = getattr(self.cfg.training, "val_every_n_epochs", 1)
        if self.val_every_n_epochs > 1:
            print(f"Validation frequency: every {self.val_every_n_epochs} epochs")
        
        # Error Handling & Recovery
        self.skip_bad_batches = getattr(self.cfg.training, "skip_bad_batches", False)
        if self.skip_bad_batches:
            print("Bad batch skipping enabled (will skip batches that cause errors)")
        
        # Training Progress Callbacks (placeholder for extensibility)
        self.callbacks = []
        
        # Automatic Best Model Loading (will be used at end of training)
        self.load_best_at_end = getattr(self.cfg.training, "load_best_at_end", True)
        
        # Training State Tracking
        self.training_state = {
            "epoch": 0,
            "global_step": 0,
            "best_val_loss": float('inf'),
            "current_val_loss": float('inf'),
        }
        
        print("=" * 80)
        print("TRAINER INITIALIZATION COMPLETE")
        print("=" * 80)

    def save_ckpt(self, is_best=False):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
                print("Created checkpoints directory")
            
            print(f"Saving checkpoint at epoch {self.epoch}...")
            ckpt = {}
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    ckpt[k] = self.__dict__[k]
            
            # Add training state to checkpoint
            ckpt['training_state'] = self.training_state
            ckpt['best_val_loss'] = self.best_val_loss
            
            latest_path = "checkpoints/model_latest.pth"
            epoch_path = f"checkpoints/model_{self.epoch}.pth"
            
            torch.save(ckpt, latest_path)
            torch.save(ckpt, epoch_path)
            
            # Save best model if this is the best
            if is_best:
                best_path = "checkpoints/model_best.pth"
                torch.save(ckpt, best_path)
                print(f" Saved new best model (val_loss={self.best_val_loss:.6f})")
            
            print(f" Checkpoint saved:")
            print(f"  - Latest: {latest_path}")
            print(f"  - Epoch {self.epoch}: {epoch_path}")
            if is_best:
                print(f"  - Best: checkpoints/model_best.pth")
            print(f"  - Saved to: {os.getcwd()}")
            
            # Clean up old checkpoints
            self.checkpoint_history.append(epoch_path)
            if len(self.checkpoint_history) > self.keep_top_k_checkpoints:
                old_ckpt = self.checkpoint_history.pop(0)
                if os.path.exists(old_ckpt) and "model_best" not in old_ckpt and "model_latest" not in old_ckpt:
                    try:
                        os.remove(old_ckpt)
                        print(f"  - Removed old checkpoint: {old_ckpt}")
                    except Exception as e:
                        print(f"  - Warning: Could not remove {old_ckpt}: {e}")
            
            # Log checkpoint save to WandB
            self.wandb_run.log({
                "checkpoint/epoch": self.epoch,
                "checkpoint/global_step": self.global_step,
                "checkpoint/saved": True,
                "checkpoint/is_best": is_best,
                "checkpoint/best_val_loss": self.best_val_loss,
            })
            
            ckpt_path = os.path.join(os.getcwd(), epoch_path)
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        print(f"Loading checkpoint from: {filename}")
        ckpt = torch.load(filename, map_location=self.device)
        for k, v in ckpt.items():
            if k not in ['training_state', 'best_val_loss']:  # Handle these separately
                self.__dict__[k] = v
        
        #  Restore training state if available
        if 'training_state' in ckpt:
            self.training_state = ckpt['training_state']
            print(f"  Restored training state: epoch={self.training_state.get('epoch', 0)}")
        if 'best_val_loss' in ckpt:
            self.best_val_loss = ckpt['best_val_loss']
            print(f"  Restored best val loss: {self.best_val_loss:.6f}")
        
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            print(f" Warning: Keys not found in checkpoint: {not_in_ckpt}")
        else:
            print(" Checkpoint loaded successfully")

    def init_models(self):
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            print(f"Resuming training from epoch {self.epoch}")
        else:
            print("No checkpoint found, starting from scratch")

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=self.encoder.emb_dim
                    + (
                        proprio_emb_dim * self.cfg.num_proprio_repeat
                        + action_emb_dim * self.cfg.num_action_repeat
                    )
                    * (self.cfg.concat_dim),
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    print(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )
        print(" Encoder, predictor, and decoder prepared with accelerator")
        
        print("Instantiating world model...")
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )
        print(" World model instantiated")

    def init_optimizers(self):
        print("Initializing optimizers...")
        
        # Always create encoder optimizer (even if not training encoder)
        print(f"  - Encoder optimizer: Adam (lr={self.cfg.training.encoder_lr})")
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.training.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        
        if self.cfg.has_predictor:
            # Always create predictor optimizer (even if not training predictor)
            print(f"  - Predictor optimizer: AdamW (lr={self.cfg.training.predictor_lr})")
            self.predictor_optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=self.cfg.training.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            print(f"  - Action/Proprio encoder optimizer: AdamW (lr={self.cfg.training.action_encoder_lr})")
            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.cfg.has_decoder:
            # Always create decoder optimizer (even if not training decoder)
            print(f"  - Decoder optimizer: Adam (lr={self.cfg.training.decoder_lr})")
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)
        
        # Learning Rate Scheduling (OPTIONAL - uncomment to enable)
        # Uncomment the following block to enable learning rate scheduling:
        # from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
        # self.schedulers = {}
        # if self.train_encoder:
        #     self.schedulers['encoder'] = CosineAnnealingLR(
        #         self.encoder_optimizer,
        #         T_max=self.total_epochs,
        #         eta_min=getattr(self.cfg.training, "min_lr", 0)
        #     )
        # if self.cfg.has_predictor and self.train_predictor:
        #     self.schedulers['predictor'] = CosineAnnealingLR(
        #         self.predictor_optimizer,
        #         T_max=self.total_epochs,
        #         eta_min=getattr(self.cfg.training, "min_lr", 0)
        #     )
        # if self.cfg.has_decoder and self.train_decoder:
        #     self.schedulers['decoder'] = CosineAnnealingLR(
        #         self.decoder_optimizer,
        #         T_max=self.total_epochs,
        #         eta_min=getattr(self.cfg.training, "min_lr", 0)
        #     )
        # self.schedulers['action_encoder'] = CosineAnnealingLR(
        #     self.action_encoder_optimizer,
        #     T_max=self.total_epochs,
        #     eta_min=getattr(self.cfg.training, "min_lr", 0)
        # )
        # print(" Learning rate schedulers initialized")
        
        print(" All optimizers initialized")

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Total epochs: {self.total_epochs}")
        print(f"Starting from epoch: {self.epoch + 1}")
        print(f"Checkpoint save frequency: every {self.cfg.training.save_every_x_epoch} epochs")
        print(f"Log every N steps: {self.log_every_steps if self.log_every_steps > 0 else 'disabled'}")
        
        # Log training start info to WandB
        if self.accelerator.is_main_process:
            self.wandb_run.log({
                "training/total_epochs": self.total_epochs,
                "training/starting_epoch": self.epoch + 1,
                "training/checkpoint_frequency": self.cfg.training.save_every_x_epoch,
                "training/log_every_steps": self.log_every_steps if self.log_every_steps > 0 else 0,
                "training/effective_batch_size": self.cfg.effective_batch_size,
                "training/batch_size_per_gpu": self.cfg.gpu_batch_size,
            })
        
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()
            print(" Planning job monitor thread started")

        init_epoch = self.epoch + 1  # epoch starts from 1
        start_time = time.time()
        
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            print("\n" + "=" * 80)
            print(f"EPOCH {self.epoch}/{init_epoch + self.total_epochs - 1}")
            print("=" * 80)
            print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            self.accelerator.wait_for_everyone()
            
            # Callback on epoch start
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_start'):
                    callback.on_epoch_start(self.epoch)
            
            self.train()
            self.accelerator.wait_for_everyone()
            
            # Configurable validation frequency
            if self.epoch % self.val_every_n_epochs == 0:
                self.val()
                self.logs_flash(step=self.epoch, mode="epoch_end")
            else:
                print(f"Skipping validation (runs every {self.val_every_n_epochs} epochs)")
            
            # Update training state
            self.training_state["epoch"] = self.epoch
            self.training_state["global_step"] = self.global_step
            if 'val_loss' in self.epoch_log:
                self.training_state["current_val_loss"] = sum(self.epoch_log['val_loss'][1]) / self.epoch_log['val_loss'][0]
            
            # Callback on epoch end
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(self.epoch, self.epoch_log)
        
            # Learning Rate Scheduling (OPTIONAL - uncomment if enabled)
            # Uncomment the following block if LR scheduling is enabled:
            # if hasattr(self, 'schedulers'):
            #     for name, scheduler in self.schedulers.items():
            #         scheduler.step()
            #         if self.accelerator.is_main_process:
            #             current_lr = scheduler.get_last_lr()[0]
            #             self.wandb_run.log({f"lr/{name}": current_lr})
            
            # Early Stopping (OPTIONAL - uncomment to enable)
            # Uncomment the following block to enable early stopping:
            # if hasattr(self, 'early_stop_patience') and self.early_stop_patience > 0:
            #     if 'val_loss' in self.epoch_log:
            #         current_val_loss = sum(self.epoch_log['val_loss'][1]) / self.epoch_log['val_loss'][0]
            #         if current_val_loss < self.best_val_loss:
            #             self.best_val_loss = current_val_loss
            #             self.patience_counter = 0
            #             print(f" New best validation loss: {self.best_val_loss:.6f}")
            #         else:
            #             self.patience_counter += 1
            #             if self.patience_counter >= self.early_stop_patience:
            #                 print(f"Early stopping triggered after {self.early_stop_patience} epochs without improvement")
            #                 print(f"Best validation loss: {self.best_val_loss:.6f}")
            #                 break
            
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch - init_epoch + 1)
            remaining_epochs = (init_epoch + self.total_epochs - 1) - epoch
            estimated_remaining = avg_time_per_epoch * remaining_epochs
            
            print(f"Epoch {self.epoch} completed in {epoch_time:.2f}s")
            print(f"Average time per epoch: {avg_time_per_epoch:.2f}s")
            print(f"Estimated remaining time: {estimated_remaining/3600:.2f} hours")
            
            # Log timing info to WandB
            if self.accelerator.is_main_process:
                self.wandb_run.log({
                    "timing/epoch_time_seconds": epoch_time,
                    "timing/avg_epoch_time_seconds": avg_time_per_epoch,
                    "timing/elapsed_time_hours": elapsed_time / 3600,
                    "timing/estimated_remaining_hours": estimated_remaining / 3600,
                    "timing/epoch": self.epoch,
                    "timing/progress_pct": (epoch - init_epoch + 1) / self.total_epochs * 100,
                })
            
            # Check if this is the best model
            is_best = False
            if 'val_loss' in self.epoch_log:
                current_val_loss = sum(self.epoch_log['val_loss'][1]) / self.epoch_log['val_loss'][0]
                self.training_state["current_val_loss"] = current_val_loss
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.training_state["best_val_loss"] = self.best_val_loss
                    is_best = True
                    print(f" New best validation loss: {self.best_val_loss:.6f}")
            
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt(is_best=is_best)
                # main thread only: launch planning jobs on the saved ckpt
                if (
                    hasattr(self.cfg, "plan_settings") and 
                    self.cfg.plan_settings.plan_cfg_path is not None
                    and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    print("Launching planning evaluation jobs...")
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)
                    print(f" Launched {len(jobs)} planning evaluation jobs")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Total epochs completed: {self.total_epochs}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("=" * 80)
        
        # Load best model for final evaluation
        if self.load_best_at_end and os.path.exists("checkpoints/model_best.pth"):
            print("\n" + "=" * 80)
            print("LOADING BEST MODEL FOR FINAL EVALUATION")
            print("=" * 80)
            self.load_ckpt("checkpoints/model_best.pth")
            print("Running final validation with best model...")
            self.val()
            self.logs_flash(step=self.epoch, mode="epoch_end")
            print("=" * 80)
        
        # Print final training state
        self.print_training_state()
        
        # Log training completion to WandB
        if self.accelerator.is_main_process:
            self.wandb_run.log({
                "training/completed": True,
                "training/total_time_hours": total_time / 3600,
                "training/total_epochs_completed": self.total_epochs,
                "training/final_epoch": self.epoch,
                "training/best_val_loss": self.best_val_loss,
            })

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        print(f"\n{'='*80}")
        print(f"TRAINING - Epoch {self.epoch}")
        print(f"{'='*80}")
        self.model.train()
        
        total_batches = len(self.dataloaders["train"])
        print(f"Total training batches: {total_batches}")
        
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train", 
                 disable=not self.accelerator.is_main_process)
        ):
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            self.model.train()
            
            # Error handling with recovery
            try:
                # Mixed precision training
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                            obs, act
                        )
                else:
                    z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                        obs, act
                    )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_bad_batches:
                    print(f" OOM error at batch {i}, skipping batch and clearing cache...")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    continue
                else:
                    raise

            # Gradient Accumulation (OPTIONAL - uncomment to enable)
            # Uncomment the following block to enable gradient accumulation:
            # if not hasattr(self, 'accumulate_grad_batches'):
            #     self.accumulate_grad_batches = 1
            #     self.accumulation_step = 0
            # 
            # self.accumulation_step += 1
            # scaled_loss = loss / self.accumulate_grad_batches

            # Zero gradients
            if self.model.train_encoder:
                self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            # Backward pass
            # Use scaled_loss if gradient accumulation is enabled
            # if hasattr(self, 'accumulate_grad_batches') and self.accumulate_grad_batches > 1:
            #     backward_loss = scaled_loss
            # else:
            backward_loss = loss
            
            # Mixed precision backward
            if self.use_amp:
                self.scaler.scale(backward_loss).backward()
            else:
                self.accelerator.backward(backward_loss)

            # FEATURE 2: Gradient Clipping (OPTIONAL - uncomment to enable)
            # Uncomment the following block to enable gradient clipping:
            # if not hasattr(self, 'gradient_clip_val'):
            #     self.gradient_clip_val = 0.0
            # 
            # if self.gradient_clip_val > 0:
            #     if self.use_amp:
            #         self.scaler.unscale_(self.encoder_optimizer)
            #     if self.model.train_encoder:
            #         torch.nn.utils.clip_grad_norm_(
            #             self.encoder.parameters(), self.gradient_clip_val
            #         )
            #     if self.cfg.has_decoder and self.model.train_decoder:
            #         torch.nn.utils.clip_grad_norm_(
            #             self.decoder.parameters(), self.gradient_clip_val
            #         )
            #     if self.cfg.has_predictor and self.model.train_predictor:
            #         torch.nn.utils.clip_grad_norm_(
            #             self.predictor.parameters(), self.gradient_clip_val
            #         )
            #         torch.nn.utils.clip_grad_norm_(
            #             list(self.action_encoder.parameters()) + list(self.proprio_encoder.parameters()),
            #             self.gradient_clip_val
            #         )

            # Optimizer steps
            # Only step when accumulation is complete
            # if hasattr(self, 'accumulate_grad_batches') and self.accumulate_grad_batches > 1:
            #     should_step = (self.accumulation_step % self.accumulate_grad_batches == 0)
            # else:
            should_step = True
            
            if should_step:
                if self.use_amp:
                    if self.model.train_encoder:
                        self.scaler.step(self.encoder_optimizer)
                    if self.cfg.has_decoder and self.model.train_decoder:
                        self.scaler.step(self.decoder_optimizer)
                    if self.cfg.has_predictor and self.model.train_predictor:
                        self.scaler.step(self.predictor_optimizer)
                        self.scaler.step(self.action_encoder_optimizer)
                    self.scaler.update()
                else:
                    if self.model.train_encoder:
                        self.encoder_optimizer.step()
                    if self.cfg.has_decoder and self.model.train_decoder:
                        self.decoder_optimizer.step()
                    if self.cfg.has_predictor and self.model.train_predictor:
                        self.predictor_optimizer.step()
                        self.action_encoder_optimizer.step()
                
                # Reset accumulation step
                # if hasattr(self, 'accumulate_grad_batches') and self.accumulate_grad_batches > 1:
                #     if self.accumulation_step % self.accumulate_grad_batches == 0:
                #         self.accumulation_step = 0

            # Memory management
            if i % self.memory_cleanup_interval == 0 and i > 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            # Gather metrics
            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            
            # Log first batch info
            if i == 0 and self.accelerator.is_main_process:
                print(f"\nFirst batch info:")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Loss components: {loss_components}")
                # Log to WandB
                wandb_log = {
                    "train/first_batch_loss": loss.item(),
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                }
                for k, v in loss_components.items():
                    wandb_log[f"train/first_batch_{k}"] = v
                self.wandb_run.log(wandb_log)
            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

            # per-step logging
            self.global_step += 1
            
            # Log to WandB more frequently (every N steps or every epoch)
            if self.accelerator.is_main_process:
                log_to_wandb = False
                if self.log_every_steps > 0 and (self.global_step % self.log_every_steps == 0):
                    log_to_wandb = True
                elif i == len(self.dataloaders["train"]) - 1:  # Last batch of epoch
                    log_to_wandb = True
                elif i % max(1, len(self.dataloaders["train"]) // 10) == 0:  # Log ~10 times per epoch
                    log_to_wandb = True
                
                if log_to_wandb:
                    # Log current batch metrics
                    step_log = {
                        "train/step_loss": loss.item(),
                        "epoch": self.epoch,
                        "global_step": self.global_step,
                        "train/batch": i,
                        "train/total_batches": len(self.dataloaders["train"]),
                    }
                    for k, v in loss_components.items():
                        step_log[f"train/step_{k}"] = v
                    
                    # Log learning rates
                    if hasattr(self, 'encoder_optimizer') and self.encoder_optimizer is not None:
                        step_log["train/lr_encoder"] = self.encoder_optimizer.param_groups[0]['lr']
                    if hasattr(self, 'predictor_optimizer') and self.predictor_optimizer is not None:
                        step_log["train/lr_predictor"] = self.predictor_optimizer.param_groups[0]['lr']
                    if hasattr(self, 'decoder_optimizer') and self.decoder_optimizer is not None:
                        step_log["train/lr_decoder"] = self.decoder_optimizer.param_groups[0]['lr']
                    if hasattr(self, 'action_encoder_optimizer') and self.action_encoder_optimizer is not None:
                        step_log["train/lr_action_encoder"] = self.action_encoder_optimizer.param_groups[0]['lr']
                    
                    self.wandb_run.log(step_log)
                    
                # Also do the regular logs_flash if configured
                if self.log_every_steps and (self.global_step % self.log_every_steps == 0):
                    self.logs_flash(step=self.epoch, mode="train")

    def val(self):
        print(f"\n{'='*80}")
        print(f"VALIDATION - Epoch {self.epoch}")
        print(f"{'='*80}")
        self.model.eval()
        
        if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            print("Running open-loop rollouts...")
            with torch.no_grad():
                train_rollout_logs = self.openloop_rollout(
                    self.train_traj_dset, mode="train"
                )
                train_rollout_logs = {
                    f"train_{k}": [v] for k, v in train_rollout_logs.items()
                }
                self.logs_update(train_rollout_logs)
                val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
                val_rollout_logs = {
                    f"val_{k}": [v] for k, v in val_rollout_logs.items()
                }
                self.logs_update(val_rollout_logs)
                
                # Log rollout metrics to WandB immediately
                if self.accelerator.is_main_process:
                    rollout_wandb_log = {
                        "epoch": self.epoch,
                        "global_step": self.global_step,
                    }
                    for k, v in train_rollout_logs.items():
                        rollout_wandb_log[f"rollout/{k}"] = v[0] if isinstance(v, list) else v
                    for k, v in val_rollout_logs.items():
                        rollout_wandb_log[f"rollout/{k}"] = v[0] if isinstance(v, list) else v
                    self.wandb_run.log(rollout_wandb_log)
            print(" Open-loop rollouts completed")

        self.accelerator.wait_for_everyone()
        total_val_batches = len(self.dataloaders["valid"])
        print(f"Total validation batches: {total_val_batches}")
        
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid",
                 disable=not self.accelerator.is_main_process)
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()

            with torch.no_grad():
                z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                    obs, act
                )

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

            # Log validation metrics to WandB
            if self.accelerator.is_main_process:
                # Log current batch validation metrics
                val_step_log = {
                    "val/step_loss": loss.item(),
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "val/batch": i,
                    "val/total_batches": len(self.dataloaders["valid"]),
                }
                for k, v in loss_components.items():
                    val_step_log[f"val/step_{k}"] = v
                
                # Log every few batches or at the end
                if i == 0 or i == len(self.dataloaders["valid"]) - 1 or i % max(1, len(self.dataloaders["valid"]) // 5) == 0:
                    self.wandb_run.log(val_step_log)

            # per-step logging
            self.global_step += 1
            if self.log_every_steps and (self.global_step % self.log_every_steps == 0):
                if self.accelerator.is_main_process:
                    self.logs_flash(step=self.epoch, mode="val")

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : 
                    start + horizon * self.cfg.frameskip + 1 : 
                    self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step, mode="train"):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step
        epoch_log["global_step"] = self.global_step

        if mode == "epoch_end":
            print(f"\n{'='*80}")
            print(f"EPOCH {step} SUMMARY")
            print(f"{'='*80}")
            if 'train_loss' in epoch_log:
                print(f"Training loss: {epoch_log['train_loss']:.6f}")
            if 'val_loss' in epoch_log:
                print(f"Validation loss: {epoch_log['val_loss']:.6f}")
            
            # Print other key metrics
            for key, value in epoch_log.items():
                if key not in ['epoch', 'global_step', 'train_loss', 'val_loss']:
                    if isinstance(value, (int, float)):
                        print(f"{key}: {value:.6f}")
            print(f"{'='*80}\n")
        elif mode == "train":
            if 'train_loss' in epoch_log:
                print(f"Step {self.global_step} - Training loss: {epoch_log['train_loss']:.6f}")
        elif mode == "val":
            if 'val_loss' in epoch_log:
                print(f"Step {self.global_step} - Validation loss: {epoch_log['val_loss']:.6f}")

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )
    
    def print_training_state(self):
        """FEATURE 12: Print current training state for debugging"""
        print("\n" + "=" * 80)
        print("TRAINING STATE SUMMARY")
        print("=" * 80)
        print(f"  Epoch: {self.training_state['epoch']}/{self.total_epochs}")
        print(f"  Global Step: {self.training_state['global_step']}")
        print(f"  Best Val Loss: {self.training_state['best_val_loss']:.6f}")
        print(f"  Current Val Loss: {self.training_state.get('current_val_loss', 'N/A')}")
        
        # Print current learning rates
        lrs = {}
        if hasattr(self, 'encoder_optimizer') and self.encoder_optimizer is not None:
            lrs['encoder'] = self.encoder_optimizer.param_groups[0]['lr']
        if hasattr(self, 'predictor_optimizer') and self.predictor_optimizer is not None:
            lrs['predictor'] = self.predictor_optimizer.param_groups[0]['lr']
        if hasattr(self, 'decoder_optimizer') and self.decoder_optimizer is not None:
            lrs['decoder'] = self.decoder_optimizer.param_groups[0]['lr']
        if hasattr(self, 'action_encoder_optimizer') and self.action_encoder_optimizer is not None:
            lrs['action_encoder'] = self.action_encoder_optimizer.param_groups[0]['lr']
        
        if lrs:
            print(f"  Current Learning Rates:")
            for name, lr in lrs.items():
                print(f"    {name}: {lr:.2e}")
        
        print("=" * 80)


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    print("\n" + "=" * 80)
    print("DINO WORLD MODEL TRAINING")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    try:
        trainer = Trainer(cfg)
        trainer.run()
        
        print("\n" + "=" * 80)
        print("TRAINING FINISHED SUCCESSFULLY")
        print("=" * 80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 80)
        raise
    except Exception as e:
        print("\n" + "=" * 80)
        print("TRAINING FAILED WITH ERROR")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
