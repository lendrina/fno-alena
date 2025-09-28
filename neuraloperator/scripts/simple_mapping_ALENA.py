from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, Dataset, Subset
import wandb
import matplotlib.pyplot as plt
import math


from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.mpu.comm import get_local_rank
from neuralop.utils import get_wandb_api_key, count_model_params


# Read the configuration
from zencfg import make_config_from_cli
import sys 
sys.path.insert(0, '../')
from neuraloperator.config.darcy_config import Default


config = make_config_from_cli(Default)
config = config.to_dict()
# Force-enable wandb logging
config["wandb"]["log_output"] = False
config["wandb"]["log"] = True
if config["wandb"]["project"] is None:
    config["wandb"]["project"] = "simple_mapping"
if config["wandb"]["entity"] is None:
    config["wandb"]["entity"] = "lendrina24" 

config["model"]["data_channels"]   = 1
config["model"]["out_channels"]    = 1
config["model"]["n_modes"]         = [32]   # 1D: number of Fourier modes
config["model"]["hidden_channels"] = 64
config["model"]["n_layers"]        = 6
config["data"]["batch_size"]       = 32
config["opt"]["training_loss"]     = "l2"   # fit-the-data stage uses L2
config["opt"]["learning_rate"]     = 1e-3
config["opt"]["scheduler"]         = "CosineAnnealingLR"
config["opt"]["scheduler_T_max"]   = config["opt"]["n_epochs"]
# Set-up distributed communication, if using
device, is_logger = setup(config)
assert torch.cuda.is_available(), "CUDA not available — install GPU PyTorch or switch runtime"
device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True  # speed-up for fixed input shapes

def log_predictions_to_wandb(model, loader, device, n_samples=3):
    model.eval()
    batch = next(iter(loader))
    x, y = batch["x"].to(device), batch["y"].to(device)
    with torch.no_grad():
        pred = model(x)

    for i in range(min(n_samples, x.shape[0])):
        plt.figure()
        plt.plot(y[i, 0].cpu().numpy(), label="Ground Truth")
        plt.plot(pred[i, 0].cpu().numpy(), label="Prediction")
        plt.legend()
        wandb.log({f"sample_{i}": wandb.Image(plt)})
        plt.close()

# Set up WandB logging
wandb_args = None
if config.wandb.log and is_logger:
    wandb.login()
    
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config.model.model_arch,
                config.model.n_layers,
                config.model.n_modes,
                config.model.hidden_channels,
            ]
        )
    wandb_args =  dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_args)

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose and is_logger:
    print(f"##### CONFIG #####\n")
    print(config)
    sys.stdout.flush()

class KSStepDatasetFromTensor(Dataset):
    def __init__(self, U: torch.Tensor, dtsave: float, target_delta: float = 0.1, start_to_use: float = 0.0, t_use: float = 0.1, drop_last_frames: int = 2):
        U = U.float()
        assert U.ndim in (3, 4), f"Expected (N,T,X) or (N,T,H,W), got {tuple(U.shape)}"
        N, T, *spatial = U.shape

        # indices (all in time steps)
        start_idx   = int(round(start_to_use / dtsave))
        stride_steps = max(1, int(round(t_use / dtsave)))
        n_ahead     = int(round(target_delta / dtsave))

        last_target = T - 1 - drop_last_frames         # last allowed target index
        last_input  = last_target - n_ahead             # last allowed input index
        if last_input < start_idx:
            raise ValueError("Not enough usable time steps after dropping last frames.")

        idxs = torch.arange(start_idx, last_input + 1, stride_steps)  # k, k+stride, ...
        # Gather pairs
        x = U[:, idxs, ...]               # (N, K, spatial...)
        y = U[:, idxs + n_ahead, ...]     # (N, K, spatial...)

        # flatten (N, K, ...) -> (N*K, ...)
        B = x.shape[0] * x.shape[1]
        x = x.reshape(B, *spatial)
        y = y.reshape(B, *spatial)

        # add channel dim
        self.X = x.unsqueeze(1).contiguous()   # (B, 1, X) or (B, 1, H, W)
        self.Y = y.unsqueeze(1).contiguous()

    def __len__(self):  return self.X.shape[0]
    def __getitem__(self, i):  return {"x": self.X[i], "y": self.Y[i]}

# Your KS file that matches the paper’s regime: ν=0.01, L=6π, dtsave=0.1
ks_path = r"C:/Users/elena/Anima's lab/temp_ac_ks/T=100,niu=0.01,N=1024,dt=0.001,6pi,dtsave=0.1,sample=200(68)._test_ut.pt"
U = torch.load(ks_path, map_location="cpu").float()    # (N, T, 1024)

# Split by trajectories so train & test share the same ν/L/dtsave (no mismatch)
N = U.shape[0]
g = torch.Generator().manual_seed(0)
perm = torch.randperm(N, generator=g)
n_train_traj = int(0.8 * N)
U_train = U[perm[:n_train_traj]]
U_test  = U[perm[n_train_traj:]]

# Build S(0.1) pairs (exact since dtsave=0.1 here)
train_dataset = KSStepDatasetFromTensor(U_train, dtsave=0.1, target_delta=0.1, start_to_use=0.0, t_use=0.1, drop_last_frames=2)
test_dataset  = KSStepDatasetFromTensor(U_test,  dtsave=0.1, target_delta=0.1, start_to_use=0.0, t_use=0.1, drop_last_frames=2)

# train_dataset = Subset(train_dataset, range(min(20000, len(train_dataset))))
# test_dataset  = Subset(test_dataset,  range(min(20000, len(test_dataset))))

train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"],
                          shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
test_loaders = {"ks_test": DataLoader(test_dataset, batch_size=config["data"]["batch_size"],
                                      shuffle=False, num_workers=0, pin_memory=True)}

# Since our dataset is already numerical tensors, no processor is required
class IdentityProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_normalizer = None
        self.out_normalizer = None
        self.device = torch.device("cuda:0")
    def forward(self, x):  
        return x
    def decode(self, x):  
        return x
    def preprocess(self, sample):
        return {"x": sample["x"].to(self.device, non_blocking=True),
                "y": sample["y"].to(self.device, non_blocking=True)}
    def postprocess(self, out, sample):
        return out, sample
    def to(self, device):
        self.device = torch.device(device)
        return self

data_processor = IdentityProcessor()

model = get_model(config).to(device)

# convert dataprocessor to an MGPatchingDataprocessor if patching levels > 0
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(model=model,
                                             in_normalizer=data_processor.in_normalizer,
                                             out_normalizer=data_processor.out_normalizer,
                                             padding_fraction=config.patching.padding,
                                             stitching=config.patching.stitching,
                                             levels=config.patching.levels,
                                             use_distributed=config.distributed.use_distributed,
                                             device=device)

# Reconfigure DataLoaders to use a DistributedSampler 
# if in distributed data parallel mode
if config.distributed.use_distributed:
    train_db = train_loader.dataset
    train_sampler = DistributedSampler(train_db, rank=get_local_rank())
    train_loader = DataLoader(dataset=train_db,
                              batch_size=config.data.batch_size,
                              sampler=train_sampler)
    for (res, loader), batch_size in zip(test_loaders.items(), config.data.test_batch_sizes):
        
        test_db = loader.dataset
        test_sampler = DistributedSampler(test_db, rank=get_local_rank())
        test_loaders[res] = DataLoader(dataset=test_db,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=test_sampler)
# Create the optimizer
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")


# Creating the losses
l2loss = LpLoss(d=1, p=2)
h1loss = H1Loss(d=1)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.training_loss} '
        f'but expected one of ["l2", "h1"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    data_processor=data_processor,
    mixed_precision=config.opt.mixed_precision,
    wandb_log=config.wandb.log,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
              )

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log, commit=False)
        wandb.watch(model)

print("Using device:", device)
print("Model is on:", next(model.parameters()).device)

# Train the model
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)
if config.wandb.log and is_logger:
    log_predictions_to_wandb(model, test_loaders["ks_test"], device)
if config.wandb.log and is_logger:
    wandb.finish()
