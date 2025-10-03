using Test
using LinearAlgebra
using MobiusSphere
using Nemo
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

        @testset "CalciumField support" begin
                C = CalciumField(extended=true)
                zeroC = C(0)
                oneC = C(1)
                _complex(x, y) = x + y * onei(C)

                infC = unsigned_infinity(C)
                MT.set_infinity(infC)

                # 0 = [0, 0, -1], 1 = [1, 0, 0], inf = [0, 0, 1]
                base_Rc = [zeroC, zeroC, -oneC]
                base_Gc = [oneC, zeroC, zeroC]
                base_Bc = [zeroC, zeroC, oneC]

                θ = const_pi(C) // 4
                rot = rotation_about_y(θ)
                # rot = [zeroC zeroC oneC;
                #         zeroC oneC zeroC;
                #         -oneC zeroC zeroC]
                tilted_Bc = rot * base_Bc
                rot_back = MobiusSphere.Btonorth(tilted_Bc)
                @test rot_back * tilted_Bc == base_Bc
                @test det(Nemo.matrix(C, rot_back)) == oneC

                # zr = C("5/4 - 1/2*I")
                zr = _complex(5 // 4, 1 // 2)
                tr = MobiusSphere.Rtozero(zr)
                @test tr == [-C(5 // 4), -C(1 // 2), zeroC]

                Gc = [C(2 // 5), C(3 // 5), C(1 // 5)]
                tr_g = MobiusSphere.Gtoone_step1(base_Bc, Gc)
                @test cross(base_Bc, tr_g) == [zeroC, zeroC, zeroC]
                shifted_G = MobiusSphere.__normalize.(Gc .+ tr_g)
                shifted_B = MobiusSphere.__normalize.(base_Bc .+ tr_g)
                local_proj = MT.stereo(tr_g)
                zg = local_proj(shifted_G)
                zb = local_proj(shifted_B)
                @test MobiusSphere.__normalize(abs(zg)) == oneC
                @test isinf(zb)

                rot_g = MobiusSphere.Gtoone_step2(_complex(3 // 5, 4 // 5))
                vec = [C(3 // 5), C(4 // 5), zeroC]
                rotated_vec = rot_g * vec
                @test rotated_vec[1] == oneC
                @test rotated_vec[2] == zeroC

                R = rot * base_Rc
                G = rot * base_Gc
                B = rot * base_Bc

                map, tr_total = MobiusSphere.Mobius_to_rigid!(R, G, B, proj)
                @test Nemo.matrix(C, map) == transpose(Nemo.matrix(C, rot))
                @test all(t -> t == zeroC, tr_total)

                projR = proj(R)
                projG = proj(G)
                projB = proj(B)

                m = MT.Mobius(projR, projG, projB) # sends 0, 1, inf -> projR, projG, projB
                # 0 = [0, 0, -1], 1 = [1, 0, 0], inf = [0, 0, 1]
                map, tr_total = MobiusSphere.Mobius_to_rigid(inv(m), [zeroC, oneC, infC])
                @test Nemo.matrix(C, map) == transpose(Nemo.matrix(C, rot))
                @test all(t -> t == zeroC, tr_total)

                map, tr_total = MobiusSphere.Mobius_to_rigid(m, [zeroC, oneC, onei(C)])
                @test Nemo.matrix(C, map) == Nemo.matrix(C, rot)
                @test all(t -> t == zeroC, tr_total)

                m2 = MobiusSphere.rigid_to_Mobius(map, tr_total, [zeroC, oneC, onei(C)])
                @test m == m2
        end
end

println("All tests passed!")
