# Option Pricing — Black-Scholes vs. Neural Networks

Can a neural network price options better than the closed-form Black-Scholes model? Both are fitted to the same 35,591 Asian Paints (NSE) option contracts from 2020 and scored against actual market closing prices.

Short answer: yes, substantially — because the network is free to learn the volatility smile and other market frictions that Black-Scholes assumes away.

## Results

Scored against actual traded closing prices (₹). Lower is better.

| Contract | Model | RMSE ↓ | MAE ↓ | MSE |
|----------|-------|--------|-------|-----|
| Call | Black-Scholes | 185.07 | 127.88 | 34,251.68 |
| Call | **MLP** | **54.57** | **38.87** | 2,978.25 |
| Put | Black-Scholes | 196.79 | 160.01 | 38,726.45 |
| Put | **MLP** | **31.64** | **20.89** | 1,001.11 |

The MLP cuts call pricing error by **3.4×** and put pricing error by **6.2×** versus the analytical model.

### Why Black-Scholes does poorly here

Black-Scholes assumes constant volatility, lognormal returns, no dividends, and frictionless continuous hedging. Indian equity options in 2020 — a year with a violent COVID volatility shock — violate essentially all of these. The network, given the same inputs, can fit the smile directly.

### An honest note on the training schedule

Both models train for 200 epochs at `lr=1e-5`, then continue fine-tuning at `1e-6`, `1e-7`, `1e-8`. **Continued fine-tuning made the call model worse, not better:**

| Stage | Call RMSE | Put RMSE |
|-------|-----------|----------|
| 200 ep @ 1e-5 | **54.57** | 41.58 |
| +20 ep @ 1e-6 | 76.44 | **31.64** |
| +10 ep @ 1e-7 | 97.56 | — |
| +5 ep @ 1e-8  | 78.67 | — |

The call model's best checkpoint is the *first* one; every later stage degrades it. Each stage recompiles the optimiser, resetting Adam's moment estimates, and there is no early stopping or best-checkpoint restore — so the reported "final" model is not the best model. The table above quotes each model's genuine best. Adding `ModelCheckpoint(save_best_only=True)` and `EarlyStopping` would be the fix.

## Method

**Features:** `t` (days to expiry), `strike_price`, `underlying_value`, `r` (risk-free rate), and a volatility estimate. Target is the option's `close` price.

**Volatility.** The Black-Scholes notebook uses the dataset's supplied `sigma`; the MLP notebooks re-estimate it as a 20-day rolling standard deviation of returns (`sigma_20`) and drop the original column. The two approaches therefore use slightly different volatility inputs — worth keeping in mind when reading the comparison as strictly like-for-like.

**Architecture.** BatchNorm on input → Dense(400) + LeakyReLU → 3 × [Dense(400) + BatchNorm + LeakyReLU] → Dense(1, relu). MSE loss, Adam, batch size 4096. The final `relu` enforces non-negative prices, which is the right inductive bias for an option premium.

**Split.** Calls and puts are modelled separately, split by moneyness. 80/20 train/test, `random_state=42`.

## Layout

```
notebooks/     final pipeline — run these in order
experiments/   earlier iterations, kept for provenance
data/          ASIANPAINT_Dataset.xlsx (35,591 contracts, 2020)
report/        WIDS project write-up
```

## Running

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Notebooks read `../data/ASIANPAINT_Dataset.xlsx` and are meant to be run from their own directory. Written for Google Colab; a GPU helps but the models are small enough to train on CPU.

## Context

Built for **WiDS (Winter in Data Science)**, IIT Bombay. `report/WIDS_project_learnings.pdf` is my own write-up of what came out of it.

## References

Neither paper is redistributed here — follow the links.

- Ke, A. & Yang, A. "Option Pricing with Deep Learning." Stanford CS229. *(the approach this project follows)*
- de Souza Santos, D. & Espínola Ferreira, T. A. "Neural Network Learning of Black-Scholes Equation for Option Pricing." [arXiv:2405.05780](https://arxiv.org/abs/2405.05780)
- Zerodha Varsity, *Module 5: Options Theory for Professional Trading* — background on Indian options markets.

## Disclaimer

An academic study of function approximation on financial data, not trading advice. The model is fitted to a single underlying in a single unusual year, and is not validated for any real pricing or trading use.
