module MobiusSphere

export Mobius_to_rigid!, Mobius_to_rigid, rigid_to_Mobius, rotation_axis_angle

include("BaseMotions.jl")

import MobiusTransformations as MT
using MobiusTransformations: MobiusTransformation, Möbius, Mobius, stereo
export MobiusTransformation, Möbius, Mobius, stereo

@inline _cross(u, v) = [u[2] * v[3] - u[3] * v[2],
                        u[3] * v[1] - u[1] * v[3],
                        u[1] * v[2] - u[2] * v[1]]

# Default: identity for plain Number types.
# MobiusSphereNemoExt adds a method for Nemo types when Nemo is loaded.
@inline __normalize(z::Number) = z
@inline __normalize(v::AbstractArray) = __normalize.(v)

# Decompose Möbius transformation into rigid motion (R, T).
# R, G, B are pre-images of 0, 1, ∞ on the unit sphere (output of stereo.(inv(m).(source))).
function Mobius_to_rigid!(R, G, B, proj)
    points = [R, G, B]

    rot1 = Btonorth(points[3])
    points = [rot1*p for p in points]

    zr = proj(points[1])
    tr1 = Rtozero(zr)
    points = [p+tr1 for p in points]

    tr2 = Gtoone_step1(points[3], points[2])
    points = [__normalize.(p+tr2) for p in points]
    temp_proj = MT.stereo(tr1+tr2)
    zg = temp_proj(points[2])

    rot2 = Gtoone_step2(zg)

    tr = rot2*(tr1 + tr2)
    map = rot2 * rot1
    return __normalize(map), __normalize(tr)
end

"""
    Mobius_to_rigid(m, source=(0, 1, 2))

Given a Möbius transformation `m` returns `Q, T` (rotation matrix and translation vector)
such that `m(z) = p_T(Q*p(z)+T)`, where `p = stereo()` is the standard stereographic
projection and `p_T = stereo(T)` is the stereo projection centred at `T`.
"""
function Mobius_to_rigid(m::MT.MobiusTransformation{T}, source=(0, 1, 2)) where {T}
    m0 = MT.Mobius(source)
    proj = MT.stereo()
    R, G, B = proj.(inv(m0*m).(source))
    return Mobius_to_rigid!(R, G, B, proj)
end

function rigid_to_Mobius(rigid_motion, source=[0, 1, 1*im])
    p = MT.stereo()
    source_sphere = p.(source)
    target_sphere = map(rigid_motion, source_sphere)
    q = MT.stereo(rigid_motion(Z(0)))
    target = q.(target_sphere)
    return MT.Möbius(source, target)
end

"""
    rigid_to_Mobius(Q, T, source=[0, 1, 1*im])

Given a 3D rotation `Q` (`Q*Q'=I`, `det(Q)=1`) and a translation vector `T`, returns
the Möbius transformation `m` defined by `m(z) = p_T(Q*p(z)+T)`.
"""
rigid_to_Mobius(Rot::AbstractMatrix, Trans::AbstractVecOrMat, source=[0, 1, 1*im]) =
    rigid_to_Mobius(pt -> Rot*pt + Trans, source)

function rotation_axis_angle(R::AbstractMatrix)
    size(R) == (3, 3) || throw(DimensionMismatch("rotation matrices must be 3×3"))
    x = R[1, 1]
    Id = I(x)
    tr = R[1, 1] + R[2, 2] + R[3, 3]
    one_x = one(x)
    cosθ = __normalize((tr - one_x) / 2)
    if cosθ isa AbstractFloat
        cosθ = clamp(cosθ, -one_x, one_x)
    end
    axis_skew = [R[3, 2] - R[2, 3], R[1, 3] - R[3, 1], R[2, 1] - R[1, 2]]
    axis_skew_sq = __normalize(axis_skew[1]^2 + axis_skew[2]^2 + axis_skew[3]^2)
    sinθ = __normalize(sqrt(axis_skew_sq) / 2)
    if sinθ isa AbstractFloat
        sinθ = clamp(sinθ, -one_x, one_x)
    end
    θ = if Base.hasmethod(atan, Tuple{typeof(sinθ), typeof(cosθ)})
        atan(sinθ, cosθ)
    else
        acos(cosθ)
    end
    if _approx_zero(θ)
        axis = [one_x, zero(x), zero(x)]
        return axis, zero(θ)
    end
    axis = nothing
    if !_approx_zero(axis_skew_sq)
        if _approx_zero(sinθ)
            norm_axis = sqrt(axis_skew_sq)
            axis = axis_skew ./ norm_axis
        else
            axis = axis_skew ./ (2 * sinθ)
        end
    else
        rows = [R[i, :] .- Id[i, :] for i in 1:3]
        for (i, j) in ((1, 2), (1, 3), (2, 3))
            candidate = _cross(rows[i], rows[j])
            if !_approx_zero(candidate[1]^2 + candidate[2]^2 + candidate[3]^2)
                norm_axis = sqrt(__normalize(candidate[1]^2 + candidate[2]^2 + candidate[3]^2))
                axis = candidate ./ norm_axis
                break
            end
        end
    end
    if axis === nothing
        axis = [one_x, zero(x), zero(x)]
    end
    return __normalize.(axis), θ
end

end # of module
