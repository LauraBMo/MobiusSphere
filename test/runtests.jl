using Test
using LinearAlgebra
using MobiusSphere
using Nemo
import MobiusTransformations as MT
using Base.MathConstants: π

const NUM_TOL = 1e-12

rotation_about_y(θ) = [cos(θ) 0 sin(θ);
                        0      1 0;
                        -sin(θ) 0 cos(θ)]

@testset "MobiusSphere" begin
    proj = MT.stereo()
    base_R = [0.0, 0.0, -1.0]
    base_G = [1.0, 0.0, 0.0]
    base_B = [0.0, 0.0, 1.0]

    @testset "Base motion primitives" begin
        θ = π/4
        tilt = rotation_about_y(θ)
        tilted_B = tilt * base_B
        rot = MobiusSphere.Btonorth(tilted_B)
        @test rot * tilted_B ≈ base_B atol=NUM_TOL
        @test isone(rot' * rot)
        @test det(rot) ≈ 1 atol=NUM_TOL

        zr = complex(1.25, -0.5)
        tr = MobiusSphere.Rtozero(zr)
        @test tr ≈ [-1.25, 0.5, 0.0] atol=NUM_TOL

        B = base_B
        G = [0.4, 0.6, 0.2]
        tr_g = MobiusSphere.Gtoone_step1(B, G)
        @test cross(B, tr_g) ≈ zeros(3) atol=NUM_TOL
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
        @test rotated_vec[2] ≈ 0 atol=NUM_TOL
        @test rotated_vec[1] ≈ 1 atol=NUM_TOL
        @test isone(rot_g' * rot_g)
    end

    @testset "Mobius ↔ rigid conversions" begin
        θ = π/4
        tilt = rotation_about_y(θ)
        R = tilt * base_R
        G = tilt * base_G
        B = tilt * base_B

        map, tr = MobiusSphere.Mobius_to_rigid!(R, G, B, proj)
        @test sum(abs2, map - rotation_about_y(-θ)) ≈ 0 atol=NUM_TOL
    end
end

println("All tests passed!")
