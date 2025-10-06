from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Dataset, Subset
import wandb
import matplotlib.pyplot as plt
import math


from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.mpu.comm import get_local_rank
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.models import SIFNO1d, WrappedSIFNO   # <-- your new model
from neuralop.losses.data_losses import ScaleConsistencyLoss

# Config system (same as simple_mapping_ALENA)
from zencfg import make_config_from_cli
sys.path.insert(0, '../')
from neuraloperator.config.darcy_config import Default

config = make_config_from_cli(Default)
config = config.to_dict()

device, is_logger = setup(config)
assert torch.cuda.is_available()
device = torch.device("cuda:0")

# Force-enable wandb logging
config["wandb"]["log_output"] = False
config["wandb"]["log"] = True
if config["wandb"]["project"] is None:
    config["wandb"]["project"] = "simple_mapping"
if config["wandb"]["entity"] is None:
    config["wandb"]["entity"] = "lendrina24" 

config["model"]["data_channels"]   = 2   # u0 + t
config["model"]["out_channels"]    = 1
config["model"]["n_modes"]         = [64]
config["model"]["hidden_channels"] = 128
config["model"]["n_layers"]        = 6
config["data"]["batch_size"]       = 64
config["opt"]["training_loss"]     = "l2"
config["opt"]["learning_rate"]     = 3e-4
config["opt"]["scheduler"]         = "CosineAnnealingLR"
config["opt"]["scheduler_T_max"]   = config["opt"]["n_epochs"]
config["opt"]["mixed_precision"] = False


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


#Dataset Wrapper for KS data with time as a scalar input
class KSTimeCondDataset(Dataset):
    def __init__(self, U: torch.Tensor, dtsave: float = 0.1, drop_last_frames: int = 2):
        U = U.float()
        N, T, X = U.shape

        last_t_idx = T - drop_last_frames
        t_idx = torch.arange(last_t_idx)
        t_vals = t_idx * dtsave
        t_max = (last_t_idx - 1) * dtsave
        t_norm = (t_vals / (t_max + 1e-8)).clamp(0, 1)

        u0 = U[:, 0, :]
        u0_rep = u0[:, None, :].repeat(1, last_t_idx, 1)
        t_rep = t_norm[None, :, None].repeat(N, 1, 1)

        self.inputs = u0_rep.reshape(-1, 1, X).contiguous()
        self.params = t_rep.reshape(-1, 1).contiguous()
        self.targets = U[:, :last_t_idx, :].reshape(-1, 1, X).contiguous()

        # helpful for debugging
        self.last_t_idx = last_t_idx
        self.t_norm = t_norm

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, i):
        return {
            "x_inputs": self.inputs[i],   # [1, X]
            "x_params": self.params[i],   # [1]   (scalar time, normalized)
            "y":        self.targets[i],  # [1, X]
        }

# Logging predictions (same style as before)
def log_predictions_to_wandb(model, loader, device, n_samples=3):
    model.eval()
    batch = next(iter(loader))
    x_inputs = batch["x_inputs"].to(device)
    x_params = batch["x_params"].to(device)
    y_true   = batch["y"].to(device)
    with torch.no_grad():
        pred = model(x_inputs=x_inputs, x_params=x_params)  # matches forward
    for i in range(min(n_samples, y_true.shape[0])):
        plt.figure()
        plt.plot(y_true[i, 0].cpu().numpy(), label="Ground Truth")
        plt.plot(pred[i, 0].cpu().numpy(), label="Prediction")
        plt.legend()
        wandb.log({f"sample_{i}": wandb.Image(plt)})
        plt.close()

# Set up WandB logging
wandb_args = None
if config["wandb"]["log"] and is_logger:
    wandb.login()
    wandb_name = config["wandb"]["name"] or "_".join(
        str(v) for v in [
            config["model"]["n_layers"],
            config["model"]["n_modes"],
            config["model"]["hidden_channels"],
        ]
    )
    wandb_args = dict(
        config=config,
        name=wandb_name,
        group=config["wandb"]["group"],
        project=config["wandb"]["project"] or "sifno-ks",
        entity=config["wandb"]["entity"] or "lendrina24",
    )
    if config["wandb"]["sweep"]:
        for key in wandb.config.keys():
            config["params"][key] = wandb.config[key]
    wandb.init(**wandb_args)

config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose and is_logger:
    print(f"##### CONFIG #####\n")
    print(config)
    sys.stdout.flush()

#data
ks_path = Path("C:/Users/elena/Anima's lab/temp_ac_ks/T=100,niu=0.01,N=1024,dt=0.001,6pi,dtsave=0.1,sample=200(68)._test_ut.pt")
ac_path = Path("C:/Users/elena/Anima's lab/temp_ac_ks/T=3000_niu=0.005_N=1024_dt=0.00625_2pi_dtsave=0.125_sample=40.pt")
U = torch.load(ac_path, map_location="cpu").float()    # (N, T, X)
N = U.shape[0]
perm = torch.randperm(N)
n_train_traj = int(0.8 * N)
U_train = U[perm[:n_train_traj]]
U_test  = U[perm[n_train_traj:]]
ac_dtsave = 0.125
ks_dtsave = 0.1
ac_frames = 0
ks_frames = 2
train_dataset = KSTimeCondDataset(U_train, dtsave=ac_dtsave, drop_last_frames=ac_frames)
test_dataset  = KSTimeCondDataset(U_test,  dtsave=ac_dtsave, drop_last_frames=ac_frames)

with torch.no_grad():
    x_mean = train_dataset.inputs.mean(dim=(0, 2), keepdim=True)
    x_std  = train_dataset.inputs.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)

    y_mean = train_dataset.targets.mean(dim=(0, 2), keepdim=True)
    y_std  = train_dataset.targets.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)

    # normalize in-place (train)
    train_dataset.inputs  = (train_dataset.inputs  - x_mean) / x_std
    train_dataset.targets = (train_dataset.targets - y_mean) / y_std

    # normalize test using *train* stats
    test_dataset.inputs   = (test_dataset.inputs  - x_mean) / x_std
    test_dataset.targets  = (test_dataset.targets - y_mean) / y_std

# Keep stats for inverse-transform / logging later
train_dataset.norm_stats = {
    "x_mean": x_mean, "x_std": x_std,
    "y_mean": y_mean, "y_std": y_std,
}

train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=config["data"]["batch_size"], shuffle=False)

test_loaders = {"ks_test": test_loader}
batch = next(iter(train_loader))


#MODEL
model = WrappedSIFNO(
    in_channels=1,
    out_channels=config["model"]["out_channels"],
    width=config["model"]["hidden_channels"],  # currently 48
    n_layers=config["model"]["n_layers"],
    n_modes=config["model"]["n_modes"][0],
    emb_channels=config["model"]["hidden_channels"],  # match width
    n_params=1
).to(device)

optimizer = AdamW(
    model.parameters(),
    lr=config["opt"]["learning_rate"],
    weight_decay=config["opt"]["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["opt"]["scheduler_T_max"]
)

# l2loss = LpLoss(d=1, p=2, reduction='sum')

# # base_loss = l2loss if config["opt"]["training_loss"] == "l2" else h1loss
# # train_loss = base_loss
# base_loss = l2loss if config["opt"]["training_loss"] == "l2" else h1loss
# train_loss = ScaleConsistencyLoss(base_loss, scales=[2,4], weight=0.1)
# eval_losses = {"h1": H1Loss(d=1), "l2": l2loss}
from neuralop.losses.data_losses import MSELoss
l2loss = LpLoss(d=1, p=2, reduction='mean')
h1loss = H1Loss(d=1, reduction='sum')
def abs_l2(pred, y):
    return l2loss.abs(pred, y)

train_loss = MSELoss()                 # absolute MSE
eval_losses = {"mse": MSELoss()}       # report absolute MSE

print("params range:", train_dataset.params.min().item(), "->",
      train_dataset.params.max().item())
batch = next(iter(train_loader))
small_dataset = Subset(train_dataset, range(128))
small_loader = DataLoader(small_dataset, batch_size=32, shuffle=True)
    
trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    mixed_precision=config.opt.mixed_precision,
    wandb_log=config.wandb.log,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
              )

if is_logger:
    n_params = count_model_params(model)
    print(f"n_params: {n_params}")
    wandb.log({"n_params": n_params})
    wandb.watch(model)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if config["wandb"]["log"] and is_logger:
    log_predictions_to_wandb(model, test_loader, device)
    wandb.finish()

