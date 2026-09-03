module MobiusSphere

export Mobius_to_rigid!, Mobius_to_rigid, Mobius_to_rigid_sitting, rigid_to_Mobius, rotation_axis_angle

include("BaseMotions.jl")

import MobiusTransformations as MT
using MobiusTransformations: MobiusTransformation, Möbius, Mobius, stereo
export MobiusTransformation, Möbius, Mobius, stereo

@inline _cross(u, v) = [u[2] * v[3] - u[3] * v[2],
                        u[3] * v[1] - u[1] * v[3],
                        u[1] * v[2] - u[2] * v[1]]

# Normalization hook. Concrete Float/Complex numbers need no normalization, so this is
# the identity; it stays as a single-method seam for a future exact-arithmetic backend.
# (The Nemo/CalciumField extension was dropped 2026-09-03 to keep the whole suite Nemo-free.)
@inline __normalize(z::Number) = z
@inline __normalize(v::AbstractArray) = __normalize.(v)
# In-place counterpart: normalize each element of `v` in place — a no-op for concrete
# numbers. `Mobius_to_rigid!` uses it to normalize its working RGB points as it moves them.
@inline __normalize!(z::Number) = z
@inline __normalize!(v::AbstractArray) = (map!(__normalize, v, v); v)

# Decompose Möbius transformation into rigid motion (R, T).
# R, G, B are pre-images of 0, 1, ∞ on the unit sphere (output of stereo.(inv(m).(source))).
function Mobius_to_rigid!(R, G, B, proj)
    RGB = [R, G, B]

    rot1 = Btonorth(RGB[3])
    RGB = Ref(rot1) .* RGB               # rotate B to the north pole

    zr = proj(RGB[1])
    tr1 = Rtozero(zr)
    RGB = RGB .+ Ref(tr1)                # slide R's projection to 0

    tr2 = Gtoone_step1(RGB[3], RGB[2])
    RGB = RGB .+ Ref(tr2)                # slide so G's projection lands on |z| = 1
    __normalize!.(RGB)

    temp_proj = stereo(tr1 + tr2)
    zg = temp_proj(RGB[2])
    rot2 = Gtoone_step2(zg)

    tr = rot2 * (tr1 + tr2)
    map = rot2 * rot1
    return __normalize(map), __normalize(tr)
end

"""
    Mobius_to_rigid(m, source=(0, 1, Inf))

Given a Möbius transformation `m` returns `Q, T` (rotation matrix and translation vector)
such that `m(z) = p_T(Q*p(z)+T)`, where `p = stereo()` is the standard stereographic
projection and `p_T = stereo(T)` is the stereo projection centred at `T`.
"""
function Mobius_to_rigid(m::MT.MobiusTransformation, source=(0, 1, Inf))
    proj = MT.stereo()
    # R, G, B are the sphere pre-images of `source` = (0, 1, ∞). With source = (0, 1, ∞)
    # the old normalising pre-map `m0 = Mobius(source)` is the identity and drops out
    # (verified: identical points to the m0 form, to machine zero).
    R, G, B = proj.(inv(m).(source))
    return Mobius_to_rigid!(R, G, B, proj)
end

"""
    Mobius_to_rigid_sitting(m, source=(0, 1, Inf))

Like [`Mobius_to_rigid`](@ref), but for the unit sphere **sitting on** the projection
plane (tangent at the south pole, centre one radius above the plane) instead of centred
on it. Returns `(Q, T)`: rotate the sphere about its centre by `Q`, translate by `T`, and
projecting the moved sphere from its top realises `m`.

This is the Arnold "sphere resting on the plane" convention. The plane's unit circle is
then the invariant circle of any `m` fixing `|z| = 1` (e.g. the accidental root-of-unity
maps) — with the centred `Mobius_to_rigid` the invariant circle sits at radius 2 instead.

The tangent-plane projection is exactly twice the centred (equatorial) one, so `m` is
conjugated by the 2× dilation (`g(z) = m(2z)/2`), decomposed on the centred sphere, and
the translation is doubled (the rotation is scale-invariant). Verified to machine precision
against the sitting-sphere caustic for the accidental maps.
"""
function Mobius_to_rigid_sitting(m::MT.MobiusTransformation, source=(0, 1, Inf))
    g_of(z) = m(2z) / 2                        # D_{1/2} ∘ m ∘ D_2  (tangent = 2× equatorial)
    zs = ComplexF64[1, im, -1]
    g  = Möbius(zs, g_of.(zs))
    Q, T = Mobius_to_rigid(g, source)
    return Q, 2 .* T
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
    axis_skew_sq = __normalize(sum(abs2, axis_skew))
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
            c2 = sum(abs2, candidate)
            if !_approx_zero(c2)
                norm_axis = sqrt(__normalize(c2))
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
