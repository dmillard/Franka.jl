# Similar to libfranka examples MotionGenerator class. Common source:
# Wisama Khalil and Etienne Dombre. 2002. Modeling, Identification and Control of Robots

ΔQ_MOTION_FINISHED = 1e-6

export JointGoalMotionGenerator
Base.@kwdef mutable struct JointGoalMotionGenerator <: AbstractRobotController{JointPositions}
    time::Float64 = 0.0
    qT::Vector{Float64}
    q0::Vector{Float64} = zeros(7)
    Δq::Vector{Float64} = zeros(7)
    dqmaxs::Vector{Float64} = zeros(7)
    t1s::Vector{Float64} = zeros(7)
    t2s::Vector{Float64} = zeros(7)
    tfs::Vector{Float64} = zeros(7)
    q1::Vector{Float64} = zeros(7)
    dqmax::Vector{Float64}
    ddqmax0::Vector{Float64}
    ddqmaxT::Vector{Float64}
end

function JointGoalMotionGenerator(speed_factor::Real, qT::Vector{Float64})
    dqmax = speed_factor * Float64[2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]
    ddqmax0 = speed_factor * Float64[5, 5, 5, 5, 5, 5, 5]
    ddqmaxT = speed_factor * Float64[5, 5, 5, 5, 5, 5, 5]
    return JointGoalMotionGenerator(; qT, dqmax, ddqmax0, ddqmaxT)
end

function _calculate_desired_values!(
    Δqd::Vector{Float64},
    c::JointGoalMotionGenerator,
    t::Float64
)
    signΔq = sign.(c.Δq)
    td = c.t2s - c.t1s
    Δt2s = c.tfs - c.t2s
    done = zeros(Bool, 7)

    for i ∈ 1:7
        Δqd[i] = if abs(c.Δq[i]) < ΔQ_MOTION_FINISHED
            done[i] = true
            0.0
        else
            if t < c.t1s[i]
                -1 / (c.t1s[i]^3) * c.dqmaxs[i] * signΔq[i] * (0.5t - c.t1s[i]) * t^3
            elseif c.t1s[i] <= t < c.t2s[i]
                c.q1[i] + ((t - c.t1s[i]) * c.dqmaxs[i] * signΔq[i])
            elseif c.t2s[i] <= t < c.tfs[i]
                c.Δq[i] + 0.5 * ((1 / (Δt2s[i]^3) * (t - c.t1s[i] - 2 * Δt2s[i] - td[i]) * (t - c.t1s[i] - td[i])^3) + (2 * t - 2 * c.t1s[i] - Δt2s[i] - 2 * td[i])) * c.dqmaxs[i] * signΔq[i]
            else
                done[i] = true
                c.Δq[i]
            end
        end
    end
    return all(done)
end

function _calculate_synchronized_values!(c::JointGoalMotionGenerator)
    dqmaxreach = copy(c.dqmax)
    tf = zeros(7)
    Δt2 = zeros(7)
    t1 = zeros(7)
    Δt2s = zeros(7)
    signΔq = sign.(c.Δq)

    for i ∈ 1:7
        if abs(c.Δq[i]) > ΔQ_MOTION_FINISHED
            if abs(c.Δq[i]) < (3 * c.dqmax[i]^2) / (4 * c.ddqmax0[i]) + (3 * c.dqmax[i]^2) / (4 * c.ddqmaxT[i])
                dqmaxreach[i] = sqrt(4 / 3 * c.Δq[i] * signΔq[i] * (c.ddqmax0[i] * c.ddqmaxT[i]) / (c.ddqmax0[i] + c.ddqmaxT[i]))
            end
            t1[i] = 1.5 * dqmaxreach[i] / c.ddqmax0[i]
            Δt2[i] = 1.5 * dqmaxreach[i] / c.ddqmaxT[i]
            tf[i] = t1[i] / 2 + Δt2[i] / 2 + abs(c.Δq[i]) / dqmaxreach[i]
        end
    end
    maxtf = maximum(tf)
    for i ∈ 1:7
        if abs(c.Δq[i]) > ΔQ_MOTION_FINISHED
            a_ = 3 / 4 * (c.ddqmaxT[i] + c.ddqmax0[i])
            b_ = -maxtf * c.ddqmaxT[i] * c.ddqmax0[i]
            c_ = abs(c.Δq[i]) * c.ddqmaxT[i] * c.ddqmax0[i]
            Δ = max(0, b_^2 - 4 * a_ * c_)
            c.dqmaxs[i] = (-b_ - sqrt(Δ)) / (2 * a_)
            c.t1s[i] = 1.5 * c.dqmaxs[i] / c.ddqmax0[i]
            Δt2s[i] = 1.5 * c.dqmaxs[i] / c.ddqmaxT[i]
            c.tfs[i] = c.t1s[i] / 2 + Δt2s[i] / 2 + abs(c.Δq[i]) / c.dqmaxs[i]
            c.t2s[i] = c.tfs[i] - Δt2s[i]
            c.q1[i] = c.dqmaxs[i] * signΔq[i] * c.t1s[i] / 2
        end
    end
end

function (c::JointGoalMotionGenerator)(state, period::Real)
    q_d0 = get_q_d(state)

    dt = period
    c.time += dt

    if c.time == 0
        c.q0 .= q_d0
        c.Δq .= c.qT - c.q0
        _calculate_synchronized_values!(c)
    end

    Δqd = zeros(7)
    motion_finished = _calculate_desired_values!(Δqd, c, c.time)

    return (c.q0 .+ Δqd, motion_finished)
end
