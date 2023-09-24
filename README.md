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
