# Compute.jl -- helper functions for PS5: Autoencoders

"""
    build_ae_model(input_dim, hidden_dim=512, latent_dim=8) -> MyAEModel

Construct an Autoencoder with a symmetric encoder-decoder architecture.

## Architecture
```
Encoder:  input_dim -> hidden_dim (relu) -> hidden_dim/2 (relu) -> latent_dim  (no activation)
Decoder:  latent_dim -> hidden_dim/2 (relu) -> hidden_dim (relu) -> input_dim  (sigmoid)
```

The bottleneck layer has no activation, leaving the latent codes unconstrained.
The sigmoid output constrains reconstructions to [0, 1], matching normalized pixel values.
"""
function build_ae_model(input_dim::Int, hidden_dim::Int=512, latent_dim::Int=8)

    h = hidden_dim ÷ 2  # half-width for the intermediate hidden layer

    # Encoder: compresses D-dim input down to the L-dim bottleneck
    enc = Chain(
        Dense(input_dim  => hidden_dim, relu),   # layer 1: input_dim -> hidden_dim with ReLU
        Dense(hidden_dim => h,          relu),   # layer 2: hidden_dim -> h with ReLU
        Dense(h          => latent_dim),          # layer 3: h -> latent_dim, no activation (free codes)
    )

    # Decoder: reconstructs D-dim output from L-dim bottleneck code
    dec = Chain(
        Dense(latent_dim => h,          relu),    # layer 1: latent_dim -> h with ReLU
        Dense(h          => hidden_dim, relu),    # layer 2: h -> hidden_dim with ReLU
        Dense(hidden_dim => input_dim,  sigmoid), # layer 3: hidden_dim -> input_dim, sigmoid clamps to [0,1]
    )

    return MyAEModel(enc, dec) # wrap encoder and decoder into a single model struct
end

"""
    load_fashion_class(class; n_examples=nothing) -> Matrix{Float32}

Load FashionMNIST training images for a single class (0-9), upscaled to 64x64.

Class labels: 0=T-shirt/top, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat,
5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot.

Returns a `(4096 x N)` Float32 matrix; each column is a flattened 64x64 image
with pixel values in [0, 1].  When `n_examples` is provided only the first
`n_examples` matching images are returned.
"""
function load_fashion_class(class::Int; n_examples::Union{Int,Nothing}=nothing)

    data = FashionMNIST(:train) # load the full FashionMNIST training split

    # find all indices in the dataset whose label matches the requested class
    idx = findall(data.targets .== class)

    # if the caller asked for fewer examples, truncate the index list
    isnothing(n_examples) || (idx = idx[1:min(n_examples, length(idx))])

    N = length(idx)

    # extract the 28x28 image tensors for the selected indices (returns 28x28xN Float32)
    imgs = data.features[:, :, idx]

    # upscale from 28x28 to 64x64 using bilinear interpolation (NNlib)
    # upsample_bilinear expects a 4D tensor: (H, W, C, N)
    imgs_4d = reshape(imgs, 28, 28, 1, N)
    imgs_up = upsample_bilinear(imgs_4d, (64 / 28, 64 / 28))  # -> (64, 64, 1, N)

    # flatten each 64x64 image into a 4096-element column vector -> (4096 x N) matrix
    return reshape(imgs_up, 4096, N)
end

"""
    show_image_grid(X; nrows=4, ncols=4, titles=nothing) -> Plots.Plot

Visualize columns of a `(D x N)` matrix as a grid of grayscale images.
The image side length is inferred from D (e.g. D=4096 -> 64x64).
If `titles` is provided it should be a vector of strings, one per image.
"""
function show_image_grid(X::AbstractMatrix; nrows::Int=4, ncols::Int=4,
                         titles::Union{Nothing, Vector{String}}=nothing)

    n = min(nrows * ncols, size(X, 2))       # how many images to display
    imgsize = round(Int, sqrt(size(X, 1)))    # infer image side length from column dimension

    # build one heatmap subplot per image using Gray colortype (matches L10d lab pattern) -
    plts = [];
    for i in 1:n
        img = Gray.(reshape(X[:, i], imgsize, imgsize)');    # reshape, transpose, convert to Gray
        t = isnothing(titles) ? "" : titles[i];               # optional per-image title
        push!(plts, heatmap(img, color=:grays, axis=false, ticks=false,
            title=t, titlefontsize=7, aspect_ratio=:equal));
    end

    # tile the subplots into a single figure with the requested grid layout -
    return plot(plts..., layout=(nrows, ncols),
        size=(max(ncols, 4) * 150, max(nrows, 1) * 150))
end
