![Python](https://img.shields.io/badge/python-3.14+-blue)
![Poetry](https://img.shields.io/badge/poetry-managed-green)
[![University of Florence](https://i.imgur.com/1NmBfH0.png)](https://ingegneria.unifi.it)

Made by [Mattia Marilli](https://github.com/mattiamarilli) and [Marco Trambusti](https://github.com/MarcoTrambusti)

# Inverse Transform Sampling with Piecewise Linear CDF

This project implements several **sampling strategies from continuous distributions** using a piecewise discretization of the cumulative distribution function (CDF).  

Currently, two main piecewise approaches are supported:

## 1. Piecewise on Equally-Spaced Points (Equispaced in x)

- **Small number of points** (e.g., <100):  
  - Sampling is done using a **single random number** and a **linear search** over the intervals.  
  - The computational cost is proportional to the number of pieces (`O(n_pieces)`).

- **Large number of points**:  
  - The **Alias Method** is used to select intervals in **constant time (`O(1)`)**.  
  - Two random numbers are used: one to select the interval via the alias table, and one to interpolate within the interval.


## 2. Piecewise on Equally-Spaced Probability Increments (Equispaced in CDF)

- The points are **equispaced in probability space**, i.e., the intervals correspond to uniform increments of the CDF.  
- Sampling uses a **single random number** to select the interval and interpolate within it, giving **constant-time performance (`O(1)`)**.  
- This approach requires knowing the inverse CDF to compute the quantile points.  

---

## Requirements

- **Python 3.14**  
- [**Poetry**](https://python-poetry.org/) for dependency management  

---

## Installation

Clone the repository and install dependencies using Poetry:

```bash
git clone https://github.com/mattiamarilli/piecewise-inverse-transform-sampling.git
cd piecewise-inverse-transform-sampling
poetry install
````

---

## Usage

Two example scripts are provided to demonstrate the sampling strategies:

1. **Exponential distribution sampling**:

```bash
poetry run python main_exp.py
```

* Generates 1,000,000 samples using:

  * Linear search on piecewise CDF
  * Alias method on piecewise CDF
  * Piecewise equispaced CDF
* Measures execution time for each method and plots histograms comparing the samples with the theoretical PDF.

2. **Gaussian distribution sampling**:

```bash
poetry run python main_gaussian.py
```

* Generates 1,000,000 samples using:

  * Linear search on piecewise CDF
  * Alias method on piecewise CDF
* Measures execution time for each method and plots histograms comparing the samples with the theoretical Gaussian PDF.