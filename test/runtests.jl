using Test
using LinearAlgebra
using MobiusSphere
import MobiusTransformations as MT
using Base.MathConstants: π

const NUM_TOL = 1e-12

rotation_about_y(θ) = [cos(θ) 0 sin(θ);
        0 1 0;
        -sin(θ) 0 cos(θ)]

@testset "MobiusSphere" begin
        proj = MT.stereo()
        # 0 = [0, 0, -1], 1 = [1, 0, 0], inf = [0, 0, 1]
        base_R = [0.0, 0.0, -1.0]
        base_G = [1.0, 0.0, 0.0]
        base_B = [0.0, 0.0, 1.0]

        @testset "Base motion primitives" begin
                θ = π / 4
                tilt = rotation_about_y(θ)
                tilted_B = tilt * base_B
                rot = MobiusSphere.Btonorth(tilted_B)
                @test rot * tilted_B ≈ base_B atol = NUM_TOL
                @test isone(rot' * rot)
                @test det(rot) ≈ 1 atol = NUM_TOL

                zr = complex(1.25, -0.5)
                tr = MobiusSphere.Rtozero(zr)
                @test tr ≈ [-1.25, 0.5, 0.0] atol = NUM_TOL

                B = base_B
                G = [0.4, 0.6, 0.2]
                tr_g = MobiusSphere.Gtoone_step1(B, G)
                @test cross(B, tr_g) ≈ zeros(3) atol = NUM_TOL
                shifted_G = MobiusSphere.__normalize.(G + tr_g)
                shifted_B = MobiusSphere.__normalize.(B + tr_g)
                local_proj = MT.stereo(tr_g)
                zg = local_proj(shifted_G)
                zb = local_proj(shifted_B)
                @test abs(abs(zg) - 1) < 1e-8
                @test isinf(zb)

                rot_g = MobiusSphere.Gtoone_step2(0.6 + 0.8im)
                vec = [0.6, 0.8, 0.0]
                rotated_vec = rot_g * vec
                @test rotated_vec[2] ≈ 0 atol = NUM_TOL
                @test rotated_vec[1] ≈ 1 atol = NUM_TOL
                @test isone(rot_g' * rot_g)
        end

        @testset "Mobius ↔ rigid conversions" begin
                θ = π / 4
                tilt = rotation_about_y(θ)
                R = tilt * base_R
                G = tilt * base_G
                B = tilt * base_B

                map, tr = MobiusSphere.Mobius_to_rigid!(R, G, B, proj)
                @test sum(abs2, map - rotation_about_y(-θ)) ≈ 0 atol = NUM_TOL
        end

        @testset "Rotation axis-angle" begin
                θ = π / 3
                rot = rotation_about_y(θ)
                axis, angle = MobiusSphere.rotation_axis_angle(rot)
                @test axis ≈ [0.0, 1.0, 0.0] atol = NUM_TOL
                @test angle ≈ θ atol = NUM_TOL

                I3 = Matrix{Float64}(I, 3, 3)
                axis_id, angle_id = MobiusSphere.rotation_axis_angle(I3)
                @test axis_id ≈ [1.0, 0.0, 0.0] atol = NUM_TOL
                @test angle_id ≈ 0.0 atol = NUM_TOL

                raw_axis = [1.0, 1.0, 1.0]
                norm_axis = raw_axis / norm(raw_axis)
                rot_pi = let x = norm_axis[1], y = norm_axis[2], z = norm_axis[3], c = cos(π), s = sin(π), v = 1 - c
                        [c + x^2 * v    x * y * v - z * s  x * z * v + y * s;
                         y * x * v + z * s  c + y^2 * v    y * z * v - x * s;
                         z * x * v - y * s  z * y * v + x * s  c + z^2 * v]
                end
                axis_pi, angle_pi = MobiusSphere.rotation_axis_angle(rot_pi)
                @test isapprox(angle_pi, π; atol = NUM_TOL)
                @test isapprox(axis_pi, norm_axis; atol = NUM_TOL) || isapprox(axis_pi, -norm_axis; atol = NUM_TOL)
        end
end

println("All tests passed!")
