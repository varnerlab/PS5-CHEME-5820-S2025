# setup paths -
const _ROOT = pwd();
const _PATH_TO_DATA = joinpath(_ROOT, "data");

# check do we have a Manifest.toml file?
using Pkg;
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages
using Statistics
using LinearAlgebra
using Distributions
using PrettyTables
using DataFrames
using CSV
using FileIO
using Flux
using NNlib

# load my codes -
# include(joinpath(_PATH_TO_SRC, "Types.jl"));
# include(joinpath(_PATH_TO_SRC, "Files.jl"));
# include(joinpath(_PATH_TO_SRC, "Compute.jl"));