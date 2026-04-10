<div align="center"> 
        
    # 📉 Optimizer Loss Landscape
    

**Visualizing how optimizers navigate loss surfaces — a geometric and intuitive exploration of deep learning optimization dynamics**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white) ![Optimization](https://img.shields.io/badge/Optimization-loss--landscape-8b5cf6) ![Research](https://img.shields.io/badge/Focus-geometry--of--training-22c55e) ![License](https://img.shields.io/badge/License-MIT-brightgreen)

</div>---

## 🔍 Overview

Training neural networks is not just about minimizing loss — it's about how different optimizers traverse the loss landscape.

This project provides a visual and experimental framework to analyze:

- How optimizers move across parameter space
- The geometry of loss surfaces
- Convergence behavior and stability
- Sharp vs flat minima

> **Why this exists: Most tutorials treat optimizers as black boxes.
> This repo makes optimization visible.**

---

## 🧠 Core Idea

``` 
Instead of only tracking scalar loss values, we project the loss surface into a 2D/3D landscape and plot optimizer trajectories over it.

Parameter Space (θ)
        │
        ▼
   Loss Function L(θ)
        │
        ▼
  Surface Sampling (grid)
        │
        ▼
  Loss Landscape Visualization
        │
        ▼
Optimizer Trajectory Overlay
``` 

---

## ⚙️ What This Project Explores

### 📉 Loss Landscape Geometry

- Smooth vs rugged surfaces
- Saddle points and local minima
- Curvature and sharpness

### 🏃 Optimizer Behavior

Compare how different optimizers move:

| Optimizer | Behavior                                  |
| --------- | ----------------------------------------- |
| SGD       | Noisy, slow but stable                    |
| Momentum  | Accelerated in consistent directions      |
| Adam      | Adaptive, fast but sometimes sharp minima |
| RMSProp   | Smooth adaptive updates                   |

---

### 🧩 Project Structure

| File                | Description                                     |
| ------------------- | ----------------------------------------------- |
| model.py            | Simple neural network (MLP / toy model)         |
| train.py            | Training loop with selectable optimizers        |
| loss_landscape.py   | Surface computation via parameter perturbations |
| plot.py             | Visualization (2D/3D contour + trajectory)      |
| optimizers.py       | Custom or wrapped optimizers                    |
| utils.py            | Helpers (sampling, normalization, tracking)     |

---

## 📐 How It Works

### 1. Train a Model

A small neural network is trained on a toy dataset.

### 2. Sample the Loss Surface

We perturb model parameters along two directions:

θ' = θ + αd₁ + βd₂

Where:

- "d₁, d₂" are random directions in parameter space
- "α, β" define a grid

### 3. Compute Loss Grid

Evaluate:

L(α, β) = Loss(θ + αd₁ + βd₂)

### 4. Plot Landscape + Trajectory

- Contour / surface plot of loss
- Overlay optimizer path during training

---

## 🚀 Quickstart

```bash
git clone https://github.com/Iamyulx/optimizer-loss-landscape.git
cd optimizer-loss-landscape
pip install -r requirements.txt
```

### Train a model:

python train.py --optimizer adam

### Generate loss landscape:

python loss_landscape.py

### Visualize:

python plot.py

---

## 📊 Example Insights

Typical observations you can reproduce:

- SGD explores wider regions → often finds flatter minima
- Adam converges faster → but sometimes sharper minima
- Momentum smooths noisy trajectories
- Flat minima often generalize better than sharp ones

---

## 🧪 Experiments to Try

- Change optimizer ("sgd", "adam", "rmsprop")
- Adjust learning rate
- Compare trajectories
- Increase model size
- Modify dataset complexity

---

## 📉 Visualization Types

- 2D contour plots
- 3D surface plots
- Trajectory overlays
- Loss vs step curves

---

## ⚠️ Limitations

This is a research/educational tool, not a production system:

This repo| Real-world training
2D projections| High-dimensional parameter space
Small models| Billion-parameter LLMs
Toy datasets| Massive real datasets
Approximate geometry| Complex curvature structures

---

## 🗺️ Roadmap

- [ ] Hessian-based curvature analysis
- [ ] Sharpness metrics (trace, eigenvalues)
- [ ] Transformer-based experiments
- [ ] Integration with real datasets
- [ ] Interactive visualization (Streamlit)
- [ ] WandB experiment tracking

---

## 📚 References

- Li et al. (2018) — Visualizing the Loss Landscape of Neural Nets
- Goodfellow et al. — Deep Learning (Optimization chapter)
- Keskar et al. (2017) — On Large-Batch Training and Sharp Minima

---

## 💡 Why This Matters

Understanding optimization at a geometric level helps you:

- Debug training instability
- Choose better optimizers
- Improve generalization
- Think like a research engineer, not just a user

---

## 📄 License

MIT © "Iamyulx" (https://github.com/Iamyulx)

---

## ⭐ If You Like This Project

Give it a star ⭐ and use it as a base for:

- Research experiments
- AI engineering portfolios
- Interview discussions (Deep Learning / Optimization)

---
