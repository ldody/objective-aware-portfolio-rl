# Objective-Aware Deep Reinforcement Learning for Portfolio Management

Research code for **objective-aware, sequential portfolio allocation** using deep reinforcement learning. The project investigates how investor-specific quantitative objectives can be incorporated directly into a portfolio decision environment and compared with an unconstrained reinforcement-learning baseline.

This repository forms part of a broader research framework connecting **natural-language investor preferences, financial information, and sequential portfolio decisions**. The associated research manuscript is currently in the publication process.

> **Portfolio note:** this repository is intended to showcase the computational methodology, software architecture, and technical skills behind the research. Empirical results from the associated manuscript are intentionally not reported here.

## Project Overview

The codebase implements a portfolio-management pipeline built around **Soft Actor-Critic (SAC)** and a custom neural actor combining convolutional, recurrent, and attention-based components.

The public implementation focuses on the quantitative portfolio-decision layer:

1. load investor-specific asset universes and quantitative objective parameters;
2. ingest and merge intraday equity data;
3. engineer market and technical-analysis features;
4. construct constrained and unconstrained portfolio environments;
5. train SAC agents using parallelized environments;
6. evaluate learned allocation policies on held-out data;
7. compare portfolio behavior using backtesting and portfolio metrics.

The associated manuscript places this implementation within a broader objective-operationalization framework in which qualitative investor guidelines are translated into quantitative portfolio objectives before being used by the decision system.

## Main Technical Skills Demonstrated

- **Deep reinforcement learning:** Soft Actor-Critic for continuous portfolio allocation
- **Deep learning:** PyTorch, CNN, GRU, Transformer encoder
- **Portfolio optimization:** dynamic asset allocation and objective-aware constraints
- **Custom RL environments:** Gymnasium-based financial environments
- **Time-series engineering:** intraday OHLCV, bid/ask information, returns and technical indicators
- **Feature engineering:** RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic RSI, SMA and EMA
- **Backtesting:** deterministic policy evaluation and portfolio-allocation tracking
- **Portfolio analytics:** returns, volatility, drawdown, turnover and related performance measures
- **Benchmarking:** constrained SAC, unconstrained SAC, and utilities for equal-weight / mean-variance strategies
- **Parallel training:** vectorized portfolio environments
- **HPC computing:** Slurm job arrays
- **Research software engineering:** modular package structure, centralized configuration and reusable training utilities

## Methodology

### 1. Investor-Specific Inputs

The repository contains two central configuration datasets:

```text
data/assets_portfolio.csv
data/parameters_portfolio.csv
```

`assets_portfolio.csv` defines the asset universe associated with each investor request/profile.

`parameters_portfolio.csv` contains quantitative portfolio objectives and their relative weights. These parameters are used by the constrained portfolio environment to influence the reinforcement-learning objective.

In the broader research framework, these structured parameters originate from the operationalization of natural-language investor guidelines. The repository therefore begins at the interface between **investor intent** and the quantitative portfolio-decision system.

### 2. Market Data and Feature Engineering

Market data are loaded from:

```text
data/raw/
```

The data pipeline is implemented under:

```text
src/drl_portfolio/data/
├── loader.py
└── features.py
```

The feature-engineering module constructs financial inputs and returns from the historical data and includes technical indicators such as:

- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Average True Range (ATR)
- Average Directional Index (ADX)
- Stochastic RSI
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)

Features are normalized with `StandardScaler` after a time-based split to preserve the temporal structure of the experiment.

### 3. Portfolio Environment

The sequential portfolio problem is implemented as a custom **Gymnasium** environment:

```text
src/drl_portfolio/envs/
├── portfolio_env.py
└── vector_env.py
```

Two environment variants are provided:

- `PortfolioEnv` — objective-aware / constrained environment;
- `PortfolioEnvNoConstraints` — baseline environment without investor-specific constraints.

At each step, the agent observes market information and the previous portfolio state and produces continuous portfolio-allocation decisions.

The vectorized environment allows multiple portfolio environments to be trained in parallel.

### 4. Soft Actor-Critic

The reinforcement-learning agent is implemented in:

```text
src/drl_portfolio/agents/sac.py
```

The project uses **Soft Actor-Critic (SAC)**, an off-policy actor-critic algorithm designed for continuous action spaces.

This is well suited to portfolio allocation because portfolio decisions are naturally represented by continuous asset weights rather than discrete trading actions.

Supporting components include:

```text
src/drl_portfolio/models/critic.py
src/drl_portfolio/utils/replay_buffer.py
```

### 5. CNN–GRU–Transformer Actor

The SAC policy uses a custom neural architecture implemented in:

```text
src/drl_portfolio/models/actor.py
```

The `CNNGRUTransformerActor` combines three representation stages:

- a **2D CNN** to extract local patterns across time and financial features;
- a **GRU** to encode temporal dependencies for each asset;
- a **Transformer encoder** to model interactions across the asset universe.

The resulting representation is combined with previous portfolio weights before producing the stochastic policy parameters used by SAC.

This architecture allows the allocation policy to jointly model temporal market structure and cross-asset relationships.

### 6. Objective-Aware Learning

The constrained environment incorporates investor-specific portfolio parameters into the decision process.

The experiment therefore compares:

```text
SAC constrained
vs.
SAC baseline (no constraints)
```

This separation makes it possible to study the effect of investor-specific objectives on the behavior of a learned portfolio policy.

The repository also contains backtesting utilities for **equal-weight** and **mean-variance** portfolio strategies. In the current research snapshot, those benchmark calls are retained in the codebase but are commented out in the main experiment routine.

### 7. Backtesting and Evaluation

Backtesting utilities are implemented in:

```text
src/drl_portfolio/utils/backtest.py
src/drl_portfolio/utils/metrics.py
```

The trained SAC policies are evaluated deterministically on the test environment.

The pipeline saves portfolio allocations and performance series to the `results/` directory, allowing downstream analysis of portfolio behavior, risk, turnover, and other evaluation criteria.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── deploy_venv.bash
├── main.py
├── data/
│   ├── assets_portfolio.csv
│   ├── parameters_portfolio.csv
│   └── raw/
│       └── <asset market-data files>
├── src/
│   └── drl_portfolio/
│       ├── agents/
│       │   └── sac.py
│       ├── data/
│       │   ├── features.py
│       │   └── loader.py
│       ├── envs/
│       │   ├── portfolio_env.py
│       │   └── vector_env.py
│       ├── models/
│       │   ├── actor.py
│       │   └── critic.py
│       ├── training/
│       │   └── train_sac.py
│       ├── utils/
│       │   ├── backtest.py
│       │   ├── metrics.py
│       │   └── replay_buffer.py
│       └── config.py
├── slurm/
│   └── main.bash
├── notebooks/
└── tests/
    └── test_load_data.py
```

## Pipeline

```text
Investor-specific parameters
           │
           ├───────────────┐
           │               │
           ▼               │
   Portfolio objectives    │
                           │
Intraday market data       │
           │               │
           ▼               │
    Feature engineering    │
           │               │
           ▼               ▼
     Time-based split + normalization
                   │
                   ▼
          Portfolio environment
             ┌─────┴─────┐
             │           │
             ▼           ▼
      Constrained SAC   SAC baseline
             │           │
             └─────┬─────┘
                   ▼
            Out-of-sample
              backtest
                   │
                   ▼
       Allocations + performance
```

## Environment

The original HPC workflow uses **Python 3.10.8**.

Create a virtual environment at the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repository requirements are:

```text
numpy
pandas
scikit-learn
gymnasium
torch
ta
```

## Running the Experiment

The main entry point is:

```bash
python main.py --task-id <TASK_ID>
```

`main.py` constructs combinations of investor requests and market-data timeframes and selects one experiment from the resulting grid using `--task-id`.

The current experiment grid contains:

```text
Investor requests: 1, 2, 3, 4, 5
Timeframes:        5min, 15min, 30min, 1h, 2h
```

This produces **25 experiment combinations**, corresponding to task IDs `0–24`.

For example:

```bash
python main.py --task-id 0
```

The selected experiment is passed to `run_full_experiment()`, which loads the corresponding data and parameters, engineers features, constructs the portfolio environments, trains the SAC agents, performs the test-period backtest, and writes outputs to `results/`.

> **Important:** the current research configuration in `main.py` uses a window of 50 timestamps and 200 training episodes. Training is computationally intensive and was originally executed in an HPC environment.

## HPC / Slurm Workflow

The repository includes:

```text
slurm/main.bash
```

The original Slurm configuration runs the complete 25-task experiment grid as a job array:

```text
#SBATCH --array=0-24%10
```

The script also contains institution-specific configuration, including cluster modules, partition settings, filesystem paths, and resource allocation.

It is preserved as part of the original research workflow and must be adapted before execution on another HPC system.

## Data

The repository snapshot contains intraday market-data CSV files for the assets used by the experimental pipeline, together with the portfolio asset and parameter tables.

The associated manuscript describes an empirical setting based on **Tokyo Stock Exchange equities** and intraday market information, with investor-specific portfolio characteristics incorporated into the decision process.

Before making the repository public, verify that you have the right to redistribute every market-data file under `data/raw/`. If the files originate from a licensed data provider, they should generally be excluded from the public repository unless redistribution is explicitly permitted.

## Reproducibility Scope

This repository preserves the research implementation used for the project rather than presenting a production-ready portfolio-management system.

Reproduction depends on:

- compatible intraday market-data files;
- the expected `assets_portfolio.csv` and `parameters_portfolio.csv` schemas;
- sufficient computational resources for SAC training;
- the original experimental assumptions embedded in the research code;
- adaptation of the Slurm configuration when running outside the original HPC environment.

The source code is intentionally presented as research software and should not be interpreted as a deployable investment system.

## Relationship to the Research Manuscript

This project contributes to the computational framework underlying the manuscript:

**“Bridging Natural Language and Forecast-Driven Financial Decision Making”**

The manuscript is **currently in the publication process**.

The broader study investigates a modular forecast-driven decision framework connecting:

1. the operationalization of natural-language investor objectives;
2. financial information and forecasting;
3. sequential portfolio decision making.

This repository primarily exposes the **deep reinforcement-learning portfolio component and its objective-aware decision environment**, together with the structured investor parameters and market-data pipeline used by that implementation.

> Results from the manuscript are intentionally not reproduced in this README while the article is in the publication process.

## Research Context

A key idea behind the project is that portfolio decisions cannot be reduced to prediction alone: the same market information can lead to different decisions depending on the objectives and constraints of the investor.

The software therefore treats investor-specific objectives as explicit inputs to a sequential decision system rather than assuming a single universal portfolio objective.

## Disclaimer

This repository is provided for **research, academic, and portfolio purposes only**.

It does not constitute investment advice, a recommendation to buy or sell securities, or a production-ready trading or portfolio-management system. Historical backtests and research experiments should not be interpreted as evidence of future investment performance.

## Author

**Léo Dody**
