# Compute.jl — helper functions for PS5: Autoencoders

"""
    build_ae_model(input_dim, hidden_dim=256, latent_dim=8) -> MyAEModel

Construct an Autoencoder with a symmetric encoder–decoder architecture.

## Architecture
```
Encoder:  input_dim → hidden_dim (relu) → hidden_dim÷2 (relu) → latent_dim  (no activation)
Decoder:  latent_dim → hidden_dim÷2 (relu) → hidden_dim (relu) → input_dim  (sigmoid)
```

The bottleneck layer has no activation, leaving the latent codes unconstrained.
The sigmoid output constrains reconstructions to [0, 1], matching normalised pixel values.
"""
function build_ae_model(input_dim::Int, hidden_dim::Int=256, latent_dim::Int=8)
    h = hidden_dim ÷ 2
    enc = Chain(
        Dense(input_dim  => hidden_dim, relu),
        Dense(hidden_dim => h,          relu),
        Dense(h          => latent_dim),        # bottleneck — no activation
    )
    dec = Chain(
        Dense(latent_dim => h,          relu),
        Dense(h          => hidden_dim, relu),
        Dense(hidden_dim => input_dim,  sigmoid),
    )
    return MyAEModel(enc, dec)
end

"""
    load_mnist_digit(digit; n_examples=nothing) -> Matrix{Float32}

Load MNIST training images for a single digit class (0–9).

Returns a `(784 × N)` Float32 matrix; each column is a flattened 28×28 image
with pixel values in [0, 1].  When `n_examples` is provided only the first
`n_examples` matching images are returned.
"""
function load_mnist_digit(digit::Int; n_examples::Union{Int,Nothing}=nothing)
    data = MNIST(:train)
    idx  = findall(data.targets .== digit)
    isnothing(n_examples) || (idx = idx[1:min(n_examples, length(idx))])
    imgs = data.features[:, :, idx]          # 28×28×N  Float32
    return reshape(imgs, 784, length(idx))   # 784×N
end

"""
    show_image_grid(X; nrows=4, ncols=4) -> Plots.Plot

Visualise columns of a `(784 × N)` matrix as a grid of 28×28 greyscale images.
"""
function show_image_grid(X::AbstractMatrix; nrows::Int=4, ncols::Int=4)
    n = min(nrows * ncols, size(X, 2))
    plts = [heatmap(reshape(Float64.(X[:, i]), 28, 28)',
                    color=:grays, axis=false, colorbar=false, aspect_ratio=1)
            for i in 1:n]
    return plot(plts..., layout=(nrows, ncols), size=(ncols * 90, nrows * 90))
end
