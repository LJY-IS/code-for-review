# DARM: Sequential Recommendation with Noisy and Biased User Behaviors

This repository provides the official implementation of DARM, a distortion-aware sequential recommendation framework that jointly models systematic bias and random noise via a mixture-of-experts architecture.

To ensure reproducibility and ease of use, we provide a Docker-based environment that allows users to run all experiments without manually configuring CUDA, PyTorch, or other dependencies.

---

## 1. Environment Requirements

- Linux system
- Docker (>= 20.10)
- NVIDIA GPU
- NVIDIA driver supporting CUDA 12.x
- NVIDIA Container Toolkit (nvidia-docker)

---

## 2. Repository Structure

```text
DARM_codes/
├── Dockerfile
└── src/
    ├── main.py
    ├── model.py
    ├── dataLoader.py
    ├── script.py
└── data/
    └── ml1m/
```

---

## 3. Build Docker Image

```bash
docker build -t darm:latest .
```

---

## 4. Run Docker Container

```bash
docker run -it --rm --gpus all -v $(pwd):/workspace darm:latest
```

---

## 5. Run Experiments

```bash
python3 ./src/main.py
```

---

## 6. Citation
