# EconPy

This repository contains my implementation of some popular econometric
estimators using PyTorch. The project is at a hobby level; hence, one should
use it cautiously.

## Currently implemented

### Linear models

#### Estimators

* OLS (and WLS)
* TSLS (2SLS) (and weighted TSLS)

#### Errors

* Constant, non-robust
* HC0 (HC - heteroskedasticity-consistent)
* HC1
* HC2
* HC3

#### Preprocessing

* Absorb
* Absorb fixed effects
* Absorb local trends
