# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PS5 for CHEME-5820 (Spring 2025): a Jupyter notebook problem set where students build and train a deterministic **Autoencoder (AE)** on FashionMNIST images (class 3 = Dress) using Julia and Flux.jl. Students implement `encode`, `decode`, and `reconstruction_loss` functions, train the model, and analyze the learned latent space.

## Environment Setup

```bash
# From the repo root, open the notebook in Jupyter/VS Code and run the first cell:
include("Include.jl")
```

`Include.jl` bootstraps everything: activates the Julia project, installs dependencies from `Project.toml` (if no `Manifest.toml` exists), loads all packages, includes source files, and initializes the `GRADER` autograder instance.

Key dependencies: **Flux.jl** (neural networks), **MLDatasets.jl** (FashionMNIST), **Plots.jl** (visualization), **JLD2** (model serialization), **OneHotArrays**, **NNlib**.

## Architecture

### Source files (`src/`)

- **Types.jl** — Defines `AbstractEncoderDecoderModel` and `MyAEModel` (mutable struct with `encoder::Chain` and `decoder::Chain` fields, registered as a Flux layer via `Flux.@layer`).
- **Compute.jl** — Helper functions: `build_ae_model(input_dim, hidden_dim=512, latent_dim=8)` constructs the symmetric encoder-decoder; `load_fashion_class(class; n_examples)` loads FashionMNIST data, upscales to 64x64, and returns a `(4096 x N)` Float32 matrix; `show_image_grid(X)` visualizes columns as grayscale image grids (image size inferred from D).
- **Autograder.jl** — `Grader` struct with `check!(grader, problem, description, pts, testfn)` for per-test scoring and `score!(grader)` for summary. Rubric: 0-4 scale where 3 = all tests pass, 4 = tests pass + discussion approved.

### Notebooks

- **PS5-CHEME-5820-Student-AE-S2025.ipynb** — Student-facing notebook with `# YOUR CODE HERE` blocks.
- **PS5-CHEME-5820-Solution-AE-S2025.ipynb** — Solution notebook.

### AE Architecture (default: D=4096, H=512, L=8)

```
Encoder: 4096 -> 512 (relu) -> 256 (relu) -> L (no activation)
Decoder: L -> 256 (relu) -> 512 (relu) -> 4096 (sigmoid)
```

Images are loaded as 28x28 from FashionMNIST and upscaled to 64x64 via `NNlib.upsample_bilinear` in `load_fashion_class`.

## Notebook Style Conventions

- Problem sets use **Tasks** (numbered sections), **callout blockquotes** (indented `>` blocks for context/instructions), **`let` blocks** for scoped computations, **"Things to think about"** or discussion questions (DQ1, DQ2, etc.), and **Key Takeaways** summaries.
- Student code is placed in a single **Implementations** section near the top, with autograder checks after each task section.
- Constants are defined in a dedicated cell after implementations.

## Autograder

Run autograder checks inline in the notebook — each `check!` call prints pass/fail with points. Call `score!(GRADER)` at the end for a summary table. Re-running a check cell updates (not duplicates) the score entry.
