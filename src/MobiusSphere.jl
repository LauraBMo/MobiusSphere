# ================ MobiusSphere.jl ================
module MobiusSphere

export Mobius_to_rigid!

# using UnPack: @unpack

using Nemo, NemoUtils

@inline _cross(u, v) = [u[2] * v[3] - u[3] * v[2],
                       u[3] * v[1] - u[1] * v[3],
                       u[1] * v[2] - u[2] * v[1]]

@inline __normalize(z::Number) = z
@inline __normalize(z) = Nemo.complex_normal_form(z)

import MobiusTransformations as MT

# For rigid transformations in 3D
# import CoordinateTransformations as CT
# import Rotations as RR

include("BaseMotions.jl")

# Decompose Möbius transformation 'm' to rigid transformation 'R, T'.
function Mobius_to_rigid!(R, G, B, proj)
    # # Map to origin sphere
    # R = proj(z0) # z0 = inv(m)(0)
    # G = proj(z1) # z1 = inv(m)(1)
    # B = proj(z∞) # z∞ = inv(m)(∞)

    # Find rigid motion 'R, T' that moves:
    #   B → north pole of target sphere
    #   R → projected to zero from traget sphere
    #   G → projected to one from target sphere

    # # Then 'R, T' correspoinds to Mobius such that
    # sends z0, z1, z∞ to 0, 1, ∞; which is 'm'.

    # Step 1: Rotate B to north pole.
    # axis_rot1 = [-B[2],B[1],0] # = cross(B_rot, SVector(0, 0, 1))
    # angle_rot1 = acos(B[3]) # = angle(B_rot, SVector(0, 0, 1))
    points = [R, G, B]
    # print("B to north\n")
    rot1 = Btonorth(points[3])
    points = [rot1*p for p in points]
    # R, G, B = [rot1].*[R, G, B]
    # all(points[3] .== [0, 0, 1]) |> println

    # print("R to zero\n")
    zr = proj(points[1])
    tr1 = Rtozero(zr)
    points = [p+tr1 for p in points]
    # R, G, B = [tr1].+[R, G, B]

    # print("G to one (step 1)\n")
    tr2 = Gtoone_step1(points[3], points[2])
    points = [__normalize.(p+tr2) for p in points]
    # R, G, B = [tr2].+[R, G, B]
    temp_proj = MT.stereo(tr1+tr2)
    zg = temp_proj(points[2])
    # println("Norm of zg: ", _norm(reim(zg)...) == 1)

    # print("G to one (step 2)\n")
    rot2 = Gtoone_step2(zg)
    points = [rot2*p for p in points]

    # map = rot2 ∘ tr2 ∘ tr1 ∘ rot1
    # print("Final Translation\n")
    tr = rot2*(tr1 + tr2)
    # print("Final Rotation")
    map = rot2 * rot1
    return map, tr
end

"""
    Mobius_to_rigid(m, source=[0, 1, 1*im])

Given a Mobius transformation `m` returns `Q, T` (`Q` rotation matrix and `T` Translation vector) such that `m(z) = p_T(Q*p(z)+T)`, where `p = stereo()` is the standard stereographic projection and `p_T = stereo(T)` stereo projection centred at `T`.
"""
function Mobius_to_rigid(m::MT.MobiusTransformation{T}, source = [0, 1, 1*im]) where T
    # # Map to origin sphere
    # R = proj(z0)
    # G = proj(z1)
    # B = proj(z∞)

    # Find rigid motion that moves:
    #   B → north pole of target sphere
    #   R → projected to zero
    #   G → projected to one
    m0 = MT.Mobius(source) # 0, 1, inf -> source
    # Now m0*m sends: zr, zg, zb -> 0, 1, inf -> source
    proj = MT.stereo()
    # R, G, B = proj.(inv(m).(inv(m0).(source)))
    R, G, B = proj.(inv(m0*m).(source))
    return Mobius_to_rigid!(R, G, B, proj)
end

function rigid_to_Mobius(rigid_motion, source=[0, 1, 1*im]) # to get Complex{Int} type.
    # We can set any source points.
    p = MT.stereo()
    source_sphere = p.(source)

    # Compute target sphere points.
    target_sphere = map(rigid_motion, source_sphere)

    # Come back to complex plane (now we are centred at T)
    q = MT.stereo(rigid_motion(Z(0)))
    target = q.(target_sphere)

    return MT.Möbius(source, target)
end

"""
    rigid_to_Mobius(Q, T, source=[0, 1, 1*im])

Given a 3D rotation `Q` (so, `Q*Q'=Id` and `det(Q)=1`) and a translation vector `T`, returns the Mobius transformation `m` corresponding to rotate the standard Riemann sphere by `Q` and translate it by `T`.
That is, the map `m` is defined as `m(z) = p_T(Q*p(z)+T)`, where `p = stereo()` is the standard stereographic projection and `p_T = stereo(T)` stereo projection centred at `T`.
"""
rigid_to_Mobius(Rot::AbstractMatrix, Trans::AbstractVecOrMat, source = [0, 1, 1*im]) =
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

# Helper function: rotation matrix from axis-angle
# function rotation_matrix(axis::AbstractVector, θ::Real)
#     u = normalize(axis)
#     ux, uy, uz = u
#     c = cos(θ)
#     s = sin(θ)
#     R = [c+ux^2*(1-c) ux*uy*(1-c)-uz*s ux*uz*(1-c)+uy*s;
#         uy*ux*(1-c)+uz*s c+uy^2*(1-c) uy*uz*(1-c)-ux*s;
#         uz*ux*(1-c)-uy*s uz*uy*(1-c)+ux*s c+uz^2*(1-c)]
#     SMatrix{3,3}(R)
# end

# # Helper function: angle between vectors
# function Base.angle(u::AbstractVector, v::AbstractVector)
#     dotprod = dot(u, v)
#     norms = norm(u) * norm(v)
#     acos(clamp(dotprod / norms, -1, 1))
# end

end # of module
