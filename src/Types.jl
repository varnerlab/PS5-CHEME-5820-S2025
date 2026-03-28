# Types.jl  -- type definitions for PS5: Autoencoders

abstract type AbstractEncoderDecoderModel end

"""
    MyAEModel <: AbstractEncoderDecoderModel

Holds the encoder and decoder networks of a deterministic Autoencoder (AE).

## Fields
- `encoder :: Chain`   -- maps input `x` (D × N) to bottleneck code `z` (L × N)
- `decoder :: Chain`   -- maps bottleneck code `z` (L × N) to reconstruction `x̂` (D × N)

## Architecture (default)
```
Encoder:  D → 256 (relu) → 128 (relu) → L   (no output activation; free bottleneck)
Decoder:  L → 128 (relu) → 256 (relu) → D   (sigmoid output, constrains x̂ to [0,1])
```

The encoder compresses each input into a low-dimensional code; the decoder attempts to
reconstruct the original from that code alone.  Training minimizes the mean-squared
reconstruction error over the training set.
"""
mutable struct MyAEModel <: AbstractEncoderDecoderModel
    encoder :: Chain
    decoder :: Chain
end

Flux.@layer MyAEModel
