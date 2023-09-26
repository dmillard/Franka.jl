<div align="center">
  <a href="https://github.com/dmillard/Franka.jl">
    <img src="docs/src/assets/logo.svg"/>
  </a>
  <p>
    1khz Franka robot control in pure Julia
  </p>
</div>

# Franka.jl

Franka.jl wraps [libfranka](https://github.com/frankaemika/libfranka) via [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl), exposing the Franka Control Interface (FCI) to Julia. It allows realtime 1khz callbacks for motion generation and gravity-compensated torque control to be written in pure Julia.

Please see the [examples/](examples/) folder for interface examples, for both motion generation and cartesian impedance control.

## Examples
|                                                                                 |
|:-------------------------------------------------------------------------------:|
| <img src="docs/src/assets/panda-julia-control.gif">                             |
| <a href="examples/y_compliant_sinusoid.jl">examples/y_compliant_sinusoid.jl</a> |


## Installation

This package is not (yet) on the Julia package repositories. In the meantime, you can install with the following steps:

1. Install a c++ compiler, Eigen3.3, and libpoco's development files
2. Clone this repo and its submodules
3. Run `path/to/cloned/Franka.jl/scripts/build.sh`
4. In Julia, run `Pkg.develop("path/to/cloned/Franka.jl")` from your calling project
