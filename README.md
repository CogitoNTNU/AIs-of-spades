# AIs of Spades

<div align="center">

![GitHub top language](https://img.shields.io/github/languages/top/CogitoNTNU/AIs-of-spades)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.1.0-blue)](https://img.shields.io/badge/version-0.1.0-blue)

</div>

A reinforcement learning system that trains a poker AI agent to play No-Limit Texas Hold'em through self-play, using the REINFORCE policy gradient algorithm.

The agent — nicknamed **UGO** — learns to decide when to fold, call, or bet, and how much to bet, by playing thousands of games against past versions of itself and optimizing for cumulative reward.

<details>
<summary><b>Table of Contents</b></summary>

- [Overview](#overview)
- [Architecture](#architecture)
  - [SimoNet (default)](#simonet-default)
  - [EvenNet (alternative)](#evennet-alternative)
- [Training Algorithm](#training-algorithm)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [CLI Arguments](#cli-arguments)
  - [Interactive UI](#interactive-ui)
- [Configuration](#configuration)
- [SLURM (HPC Cluster)](#slurm-hpc-cluster)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Team](#team)
- [License](#license)

</details>

---

## Overview

AIs of Spades trains a poker agent using the following pipeline:

1. **Simulation** — A pool of worker processes runs parallel poker games. Each worker plays UGO against sampled opponents from a checkpoint pool (past versions of itself).
2. **Learning** — The main process collects all trajectories, computes the REINFORCE loss on GPU, and updates the network.
3. **Checkpointing** — The updated model is saved periodically and added to the opponent pool, so future games are played against an increasingly diverse set of strategies.

Key features:
- **Self-play with population diversity** — opponents are sampled from the full history of checkpoints, not just the latest version
- **Curriculum learning** — showdown rewards are amplified early in training to teach card strength, then decay as strategic play becomes relevant
- **Per-action baselines** — separate rolling baselines for fold/call/bet reduce gradient variance without a separate value network
- **Structural bonuses** — reward shaping for eliminations, survival, steals, and showdowns, none of which encode direct knowledge of hand strength

---

## Architecture

Two neural network architectures are implemented, both subclassing `PokerNet`.

All models share the same output interface:
- `action_logits` — `[B, 3]` raw logits for FOLD / CALL / BET
- `bet_mean` — `[B, 1]` normalized bet fraction ∈ (0, 1) via sigmoid
- `bet_std` — `[B, 1]` standard deviation for the bet distribution via exp+clamp

### SimoNet (default)

A Transformer-based model with recurrent cross-hand state. Default and recommended.

```
Input Encoders (each produces a d_model-dim token)
├── CardsEncoder         — hole + board cards via self-attention
├── BetsNN               — 64×8 action history via MLP
├── ObsNN                — 10 scalar features (stacks, pot, street, ...)
├── hand_state_proj      — per-hand recurrent state (reset each hand)
├── game_state_proj      — cross-hand recurrent state (persists the session)
└── opp_proj × 5         — one token per opponent (randomly shuffled)
         ↓
TransformerTrunk
├── All tokens + learnable CLS token
├── Multi-head self-attention (4 heads, 2 layers)
└── Returns CLS hidden state
         ↓
Output Heads (from CLS)
├── policy_head          → [3] action logits
├── bet_mean_head        → [1] bet mean (sigmoid)
├── bet_std_head         → [1] bet std (exp + clamp to [1e-4, 1.0])
├── hand_state_head      → updated hand state
└── game_state_head      → updated game state
```

Key hyperparameters (defaults):

| Parameter | Value | Description |
|---|---|---|
| `d_model` | 64 | Embedding dimension throughout |
| `trunk_heads` | 4 | Attention heads in main trunk |
| `trunk_layers` | 2 | Transformer layers in main trunk |
| `cards_heads` | 4 | Attention heads in card encoder |
| `hand_state_dim` | 64 | Per-hand recurrent state size |
| `game_state_dim` | 64 | Cross-hand recurrent state size |
| `shuffle_opponents` | True | Randomly permute opponent tokens |

### EvenNet (alternative)

A multi-branch CNN architecture. Lighter and simpler, but less expressive.

```
Branches (processed in parallel)
├── CardsCNN     — convolutional encoder on [B, 1, 5, 13] card grid
├── BetsNN       — MLP on flattened action history
└── StateFusionNN — MLP on hand + game state
         ↓
Concatenate → MLP trunk → Output heads
```

---

## Training Algorithm

UGO is trained with **REINFORCE** with per-action baselines and entropy regularization.

**Loss components:**

```
disc_loss   = mean( -advantage × log_prob(action) )

cont_loss   = mean( -advantage × log_prob(bet_amount) )   # BET steps only
            clamped to [-5, 5] per step

entropy_bonus    = -entropy_coef × H(π)                   # maximize entropy

diversity_penalty = coef × Σ relu(mean_p - hi) + relu(lo - mean_p)
                                                           # fires when mean action prob
                                                           # exits [lo, hi] band

total_loss = disc_loss + continuous_weight × cont_loss
           + diversity_penalty + entropy_bonus
```

**Advantages:**

```
raw_advantage  = reward - baseline[action_type]
advantage      = (raw_advantage - mean) / (std + ε)       # globally normalized

baseline[a]    = rolling mean of rewards for action a
                 over the last action_baseline_window epochs
```

**Gradient flow:**
- Backward pass on GPU
- Gradient clipping to L2 norm ≤ `grad_clip_norm`
- Adam optimizer with StepLR decay

**Curriculum:**

The `showdown_reward_multiplier` linearly decays from `2.0` → `1.0` over `30000` epochs, amplifying card-strength signal early in training.

`games_per_epoch` and `hands_per_game` can be scheduled as milestone dictionaries to gradually increase data complexity.

---

## Prerequisites

- **Python 3.12** — pinned; some compiled deps don't build on 3.14+
- **UV** — dependency manager. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **Git**

Optional:
- **CUDA-capable GPU** — required for the GPU SLURM script; CPU training is supported but slower
- **SLURM cluster** — for HPC job submission

---

## Installation

```bash
git clone https://github.com/CogitoNTNU/AIs-of-spades.git
cd AIs-of-spades

# Base dependencies only
uv sync

# With Weights & Biases experiment tracking (recommended for training)
uv sync --group train

# All groups (base + training + dev + docs)
uv sync --all-groups
```

For development, install pre-commit hooks:

```bash
uv run pre-commit install
```

---

## Usage

### Training

Start a new training run:

```bash
uv run python src/main.py --run-name my-run
```

Resume from the latest checkpoint:

```bash
uv run python src/main.py --run-name my-run --resume-latest
```

Resume from a specific checkpoint:

```bash
uv run python src/main.py --resume-from res/checkpoints/my-run/SimoNet/epoch_1000.pt
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--run-name` | str | None | Sets wandb run name and checkpoint dir to `res/checkpoints/<name>` |
| `--resume-latest` | flag | False | Resume from the latest checkpoint in the checkpoint directory |
| `--resume-from` | str | None | Path to a specific checkpoint file |
| `--num-workers` | int | None | Override `num_workers` from config (set automatically by SLURM) |
| `--config-file` | str | `config.yaml` | Path to config file, relative to `src/` |

If both `--resume-latest` and `--resume-from` are set, `--resume-from` takes priority.

### Interactive UI

To play against a trained model in the browser:

```bash
uv run python src/start_ui_host.py
```

See `src/README.md` for UI configuration details.

---

## Configuration

All hyperparameters are in `src/config.yaml`.

```yaml
weight_manager:
  model_class: "SimoNet"       # "SimoNet" or "EvenNet"
  max_models: 100              # Max checkpoints to keep on disk
  keep_latest: 20              # Always retain the N most recent checkpoints
  sampling_mode: "linear"      # Opponent sampling: "uniform", "linear", or "exponential"
  checkpoint_dir: "res/..."    # Overridden by --run-name at runtime

learning_loop:
  learning_rate: 3e-4
  lr_decay_steps: 100
  lr_decay_gamma: 0.97
  epochs: 100000
  save_interval: 100
  num_workers: 8               # Parallel game workers (overridden by --num-workers)
  grad_clip_norm: 1.0
  continuous_weight: 0.01      # Weight of bet-size loss relative to action loss
  entropy_coef: 0.05           # Policy entropy regularization
  diversity_coef: 0.15         # Anti-collapse penalty coefficient
  diversity_lo: 0.08           # Lower bound of acceptable action probability
  diversity_hi: 0.80           # Upper bound of acceptable action probability
  action_baseline_window: 50   # Rolling window for per-action baselines

  # Both fields support milestone scheduling: { epoch: value, ... }
  games_per_epoch:
    0:     5
    500:   10
    1000:  20
    5000:  30
    10000: 50

  hands_per_game:
    0:     70
    300:   50
    1000:  40
    5000:  30
    10000: 25

game_loop:
  decadiment_factor: 0.8           # Discount (γ) across hands within a game
  stack_lo: 50.0                   # Min starting stack (in big blinds)
  stack_hi: 200.0                  # Max starting stack

  # Structural bonuses — all zero-knowledge (no card-strength info encoded)
  elimination_bonus: 40.0          # Per opponent eliminated
  survival_bonus:   200.0          # When UGO is last player standing
  isolation_bonus:    5.0          # For playing heads-up
  showdown_bonus:     2.0          # For reaching showdown
  steal_bonus:        8.0          # For winning an uncontested pot

  # Curriculum: showdown reward multiplier decays start → end over N epochs
  showdown_reward_multiplier_start:  2.0
  showdown_reward_multiplier_end:    1.0
  showdown_reward_multiplier_epochs: 30000
```

**Opponent sampling modes:**

| Mode | Behavior |
|---|---|
| `uniform` | Equal probability for all checkpoints |
| `linear` | Weight ∝ epoch number (bias toward recent) |
| `exponential` | Weight ∝ 2^epoch (strong recency bias) |

---

## SLURM (HPC Cluster)

Two scripts are provided for NTNU's Idun/EPIC cluster.

### GPU training (`training.slurm`)

```
Time limit:    4 hours
Partition:     GPUQ
GPUs:          1
CPUs:          14 (12 workers + 2 for main process)
Memory:        32 GB
```

### CPU training (`trainingCPUs.slurm`)

```
Time limit:    100 hours
Partition:     default (CPU nodes)
CPUs:          6 (4 workers + 2 for main process)
Memory:        50 GB
```

### Naming a run

Both scripts read the `JOB_NAME` environment variable to set the job name, output directory, and checkpoint path simultaneously:

```bash
JOB_NAME="ugo-v2" sbatch trainingCPUs.slurm
JOB_NAME="ugo-v2" sbatch training.slurm
```

This will:
- Write logs to `slurm_outputs/ugo-v2/output.txt` and `output.err`
- Save checkpoints to `res/checkpoints/ugo-v2/`
- Name the wandb run `ugo-v2`
- Rename the SLURM job to `ugo-v2` via `scontrol update`

Without `JOB_NAME`, the default name `ais-of-spades` is used.

The scripts automatically resume from the latest checkpoint (`--resume-latest`) and set `--num-workers` based on the allocated CPUs minus 2.

---

## Project Structure

```
AIs-of-spades/
├── src/
│   ├── main.py                    # Entry point
│   ├── config.yaml                # All hyperparameters
│   ├── nn/
│   │   ├── poker_net.py           # Abstract base class for all models
│   │   ├── simo_net/              # SimoNet (Transformer-based, default)
│   │   └── even_net/              # EvenNet (CNN-based, alternative)
│   ├── training/
│   │   ├── learning_loop.py       # Training loop, loss, optimizer
│   │   ├── game_loop.py           # Game simulation, reward shaping
│   │   ├── weight_manager.py      # Checkpoint pool & opponent sampling
│   │   ├── player_agent.py        # Agent wrapper
│   │   └── wandb_compat.py        # Weights & Biases integration
│   ├── pokerenv/                  # No-Limit Texas Hold'em environment
│   │   ├── table.py               # Main Gym environment
│   │   ├── player.py              # Player class
│   │   ├── observation.py         # Observation data structures
│   │   ├── common.py              # PlayerAction, PlayerState enums
│   │   └── table_engine/          # Betting, pot, and street logic
│   └── ui/                        # Browser-based interactive UI
├── training.slurm                 # SLURM script — GPU
├── trainingCPUs.slurm             # SLURM script — CPU only
├── tests/                         # Test suite
└── docs/                          # Documentation source
```

---

## Testing

```bash
uv run pytest --doctest-modules --cov=src --cov-report=html
```

### Documentation site

```bash
uv run mkdocs build
uv run mkdocs serve        # Preview at http://127.0.0.1:8000/
```

---

## Team

This project is developed by [Cogito NTNU](https://github.com/CogitoNTNU).

<table align="center">
    <tr>
        <td align="center">
            <a href="https://github.com/Simooo45">
              <img src="https://github.com/Simooo45.png?size=100" width="100px;" alt="Simooo45"/><br />
              <sub><b>Simooo45</b></sub>
            </a>
        </td>
    </tr>
</table>

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
