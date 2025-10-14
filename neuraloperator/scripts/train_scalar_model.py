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
from neuralop.models import WrappedSIFNO   # <-- new model
from neuralop.losses.data_losses import ScaleConsistencyLoss


from zencfg import make_config_from_cli
sys.path.insert(0, '../')
from neuraloperator.config.darcy_config import Default

config = make_config_from_cli(Default)
config = config.to_dict()

device, is_logger = setup(config)
# assert torch.cuda.is_available()
# device = torch.device("cuda:0")

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


class TimeShiftSemigroupDataset(Dataset):
    def __init__(self, U, dtsave,
                 min_h_frames=1, max_h_frames=None,
                 drop_tail_frames=0, n_params=1,
                 norm_stats=None, apply_norm=False):
        super().__init__()
        assert U.ndim >= 3, "U must be [N, T, *S]"
        self.U = U.float()
        self.N, self.T = U.shape[:2]
        # Detect true spatial shape (ignore leading singleton dims)
        self.S = tuple(s for s in U.shape[2:] if s > 1)
        self.D = len(self.S)
        self.dtsave = float(dtsave)
        self.drop_tail = int(drop_tail_frames)
        self.valid_T = self.T - self.drop_tail
        self.n_params = int(n_params)

        self.min_h = max(1, int(min_h_frames))
        self.max_h = self.valid_T - 1 if max_h_frames is None else int(max_h_frames)
        self.max_h = max(self.min_h, self.max_h)

        self.index = [(n, k) for n in range(self.N)
                              for k in range(self.valid_T - self.min_h)]

        self.apply_norm = bool(apply_norm)
        if norm_stats is None:
            self.x_mean = self.x_std = self.y_mean = self.y_std = None
        else:
            self.x_mean = norm_stats["x_mean"]
            self.x_std  = norm_stats["x_std"]
            self.y_mean = norm_stats["y_mean"]
            self.y_std  = norm_stats["y_std"]

    def __len__(self):
        return len(self.index)

    def _sample_delta(self):
        return torch.randint(self.min_h, self.max_h + 1, (1,), dtype=torch.long).item()

    def __getitem__(self, idx):
        n, k = self.index[idx]
        Δ = self._sample_delta()
        k2 = min(k + Δ, self.valid_T - 1)

        u_raw = self.U[n, k]
        y_raw = self.U[n, k2]
        u_sp = u_raw.squeeze()
        y_sp = y_raw.squeeze()

        if y_sp.ndim != u_sp.ndim:
            while y_sp.ndim < u_sp.ndim:
                y_sp = y_sp.unsqueeze(0)
            while u_sp.ndim < y_sp.ndim:
                u_sp = u_sp.unsqueeze(0)

        u = u_sp.reshape(self.S).unsqueeze(0)  # [1, *S]
        y = y_sp.reshape(self.S).unsqueeze(0)  # [1, *S]

        if self.apply_norm and (self.x_mean is not None):
            eps = 1e-6
            u = (u - self.x_mean) / self.x_std.clamp_min(eps)
            y = (y - self.y_mean) / self.y_std.clamp_min(eps)

        h = float(Δ) * self.dtsave
        params = torch.zeros(self.n_params, dtype=u.dtype)
        params[0] = h

        return {"x_inputs": u, "x_params": params, "y": y}

def compute_norm_stats(U_train):
    # U_train: [N, T, *S]
    m = U_train.mean(dim=(0, 1))                     # [*S]
    s = U_train.std(dim=(0, 1)).clamp_min(1e-6)      # [*S]
    m = m.unsqueeze(0)                                # [1, *S]
    s = s.unsqueeze(0)                                # [1, *S]
    return {"x_mean": m, "x_std": s, "y_mean": m, "y_std": s}

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
ks_path = Path("C:/Users/elena/fno-alena/temp_ac_ks/T=100,niu=0.01,N=1024,dt=0.001,6pi,dtsave=0.1,sample=200(68)._test_ut.pt")
ac_path = Path("C:/Users/elena/fno-alena/temp_ac_ks/T=3000_niu=0.005_N=1024_dt=0.00625_2pi_dtsave=0.125_sample=40.pt")
U = torch.load(ac_path, map_location="cpu").float()
N = U.shape[0]
perm = torch.randperm(N)
n_train_traj = int(0.8 * N)
U_train = U[perm[:n_train_traj]]
U_test  = U[perm[n_train_traj:]]
ac_dtsave = 0.125
ks_dtsave = 0.1
ac_frames = 0
ks_frames = 2
n_params = 1  # or >1 if you want vector params later
norm_stats = compute_norm_stats(U_train)
train_dataset = TimeShiftSemigroupDataset(
    U_train, dtsave=ac_dtsave, min_h_frames=1, max_h_frames=None,
    drop_tail_frames=0, n_params=n_params, norm_stats=norm_stats, apply_norm=True
)
test_dataset  = TimeShiftSemigroupDataset(
    U_test,  dtsave=ac_dtsave, min_h_frames=1, max_h_frames=None,
    drop_tail_frames=0, n_params=n_params, norm_stats=norm_stats, apply_norm=True
)

train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=config["data"]["batch_size"], shuffle=False)

test_loaders = {"ks_test": test_loader}
batch = next(iter(train_loader))
b0 = next(iter(train_loader))
spatial = tuple(b0["x_inputs"].shape[2:])

#MODEL
model = WrappedSIFNO(
    in_channels=1,
    out_channels=1,
    width=128,
    n_layers=6,
    n_modes=64,
    emb_channels=128,
    n_params=n_params
).to(device)

model.sifno._build(spatial)
model = model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=config["opt"]["learning_rate"],
    weight_decay=config["opt"]["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["opt"]["scheduler_T_max"]
)

b = next(iter(train_loader))
print("x_inputs", b["x_inputs"].shape)
print("x_params", b["x_params"].shape)
print("y",       b["y"].shape)

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

train_loss = MSELoss() #absolute MSE
eval_losses = {"mse": MSELoss()}
    
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

