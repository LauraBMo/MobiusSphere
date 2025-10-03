# ================ MobiusSphere.jl ================
module MobiusSphere

export Mobius_to_rigid!

# using UnPack: @unpack

using Nemo, NemoUtils

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
    #   R → projected to zero
    #   G → projected to one

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

function Mobius_to_rigid(m::MT.MobiusTransformation{T}) where T
    # # Map to origin sphere
    # R = proj(z0)
    # G = proj(z1)
    # B = proj(z∞)

    # Find rigid motion that moves:
    #   B → north pole of target sphere
    #   R → projected to zero
    #   G → projected to one
    R, G, B = holytrinity(inv(m))
    return Mobius_to_rigid!(R, G, B, proj)
end

function rigid_to_Mobius(Rot, Trans)
    base_points = holytrinity()
    im_points = ([Rot].*base_points) .+ [Trans]
    im_ht = stereo(Trans).(im_points)
    return MT.Mobius(im_ht)
end

# 3D image of the standard base points.

__holytrinity() = [0, 1, MT.infinity()]

holytrinity(m::MT.MobiusTransformation,
            proj::MT.StereographicProjection) = proj.(m.(__holytrinity()))

holytrinity(m::MT.MobiusTransformation) = holytrinity(m, stereo())
holytrinity(proj::MT.StereographicProjection) = proj(__holytrinity())
holytrinity() = holytrinity(stereo())

# # Helper function: rotation matrix from axis-angle
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
