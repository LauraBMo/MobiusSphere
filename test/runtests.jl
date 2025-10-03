# runtests.jl
using Test
using LinearAlgebra
# using StaticArrays
using MobiusSphere
using Nemo
import MobiusTransformations as MT

# Tolerance constants
const NUM_TOL = 1e-12
const SPHERE_TOL = 1e-8

@testset "MobiusSphere Tests" begin

    # Standard Riemann sphere
    proj_s1 = MobiusSphere.StereographicProjection(s1)

    @testset "Core Functionality" begin
        @testset "Sphere Construction" begin
            @test s1.center == [0,0,0]

            s2 = MobiusSphere.Sphere([1,0,0])
            @test norm([1,0,1] - s2.center) ≈ 1 atol=NUM_TOL
        end

        @testset "Stereographic Projections" begin
            # Test round trips
            points = [
                [0,0,-1], [1,0,0], [0,1,0], [-1,0,0],
                [0,-1,0], [0.5,0.5,sqrt(0.5)], [0.3,-0.4,sqrt(1-0.09-0.16)]
            ]
            for P in points
                z = proj_s1(P)
                P_back = proj_s1(z)
                @test norm(P - P_back) < NUM_TOL
            end

            # Test infinity
            @test proj_s1([0,0,1]) == complex(Inf)
            @test proj_s1(complex(Inf)) ≈ [0,0,1] atol=NUM_TOL
        end

        @testset "Elementary Rigid Transformations" begin
            id = MobiusSphere.identity_transformation()
            P = [1,2,3]
            @test id(P) ≈ P

            # Horizontal translation
            t = complex(1.5, -2.0)
            ht = MobiusSphere.horizontal_translation(t)
            @test ht(P) ≈ P + [real(t), imag(t), 0]

            # Vertical translation
            vt = MobiusSphere.vertical_translation(3.0)
            @test vt(P) ≈ P + [0,0,3.0]

            # Rotation composition
            rz = MobiusSphere.rotation_about_z(π/3)
            rx = MobiusSphere.rotation_about_x(π/4)
            comp = rx * rz
            @test comp(P) ≈ rx(rz(P))
        end
    end

    @testset "BaseMotions Edge Cases" begin
        B_south = [0.0, 0.0, -1.0]
        rot = MobiusSphere.Btonorth(B_south)
        @test rot * B_south ≈ [0.0, 0.0, 1.0] atol=NUM_TOL
        I3 = Matrix{eltype(rot)}(I, 3, 3)
        @test rot' * rot ≈ I3 atol=NUM_TOL

        tr = MobiusSphere.Gtoone_step1(B_south, B_south)
        @test tr ≈ zeros(3) atol=NUM_TOL

        @testset "Gtoone_step2 Rotation" begin
            rot = MobiusSphere.Gtoone_step2(complex(0.6, -0.8))
            P = [0.3, -0.4, 0.5]
            rotated = rot * P
            rot_I = Matrix{eltype(rot)}(I, 3, 3)
            @test rot' * rot ≈ rot_I atol=NUM_TOL
            @test abs(rotated[3]) ≈ abs(P[3]) atol=NUM_TOL
        end
    end

    @testset "Mobius to Rigid Transformation" begin

        @testset "Automatic Seed Evaluation" begin
            s = MobiusSphere.Sphere(Float64)
            proj = MobiusSphere.StereographicProjection(s)
            m = MobiusSphere.MobiusTransformation(2.0, 1.0 + 2.0im, 3.0, 4.0)

            z0 = zero(Float64)
            z1 = one(Float64)
            z∞ = MT.infinity(Float64)

            R = proj(m(z0))
            G = proj(m(z1))
            B = proj(m(z∞))

            manual = MobiusSphere.Mobius_to_rigid!(R, G, B, proj)
            automatic = MobiusSphere.Mobius_to_rigid!(m, proj)

            @test manual == automatic
        end

        @testset "Automatic Seed Evaluation with CalciumField" begin
            K = CalciumField()
            a = K(2)
            b = K(1)
            c = K(3)
            d = K(4)

            mK = MobiusSphere.MobiusTransformation(a, b, c, d)
            center = [zero(a), zero(a), zero(a)]
            sK = MobiusSphere.Sphere(center)
            projK = MobiusSphere.StereographicProjection(sK)

            z0 = zero(a)
            z1 = one(a)
            z∞ = MT.infinity(typeof(z0))

            Rk = projK(mK(z0))
            Gk = projK(mK(z1))
            Bk = projK(mK(z∞))

            manualK = MobiusSphere.Mobius_to_rigid!(Rk, Gk, Bk, projK)
            automaticK = MobiusSphere.Mobius_to_rigid!(mK, projK)

            @test manualK == automaticK
        end

        @testset "Decomposition Verification" begin
            # Test case that previously failed
            m = MobiusSphere.MobiusTransformation(2.0, 1.0+2.0im, 3.0, 0.0)
            rt = MobiusSphere.decompose_to_rigid(m)

            # Test specific point
            z = complex(1.5, -0.5)
            expected = (2*z + (1+2im)) / (3*z)

            s1 = MobiusSphere.Sphere(Float64)
            proj_s1 = MobiusSphere.StereographicProjection(s1)
            s2 = rt(s1)
            proj_s2 = MobiusSphere.StereographicProjection(s2)
            # @test proj_s2(np(s2)) == m(complex(Inf))

            P = proj_s1(z)
            P_rt = rt(P)
            @test norm(P_rt - s2.center) ≈ 1
            z_rt = proj_s2(P_rt)

            @test abs(z_rt - expected) < NUM_TOL
        end

        @testset "Special Case (c=0)" begin
            # Test case: m(z) = 2z + (1+2im)
            m = MobiusSphere.MobiusTransformation(2.0, 1.0+2.0im, 0.0, 1.0)
            rt = MobiusSphere.decompose_to_rigid(m)
            s2 = rt(s1)  # Target sphere
            proj_s2 = MobiusSphere.StereographicProjection(s2)

            # Test north pole mapping
            np_s2 = np(s2)
            np_proj = proj_s2(np_s2)
            @test isinf(np_proj)
            @test m(complex(Inf)) ≈ complex(Inf)

            # Test point consistency
            z = complex(1.5, -0.5)
            P = proj_s1(z)
            P_rt = rt(P)
            @test norm(P_rt - s2.center) ≈ 1 atol=SPHERE_TOL

            z_rt = proj_s2(P_rt)
            @test abs(z_rt - m(z)) < NUM_TOL
        end

        @testset "General Case (c≠0)" begin
            # Test case: m(z) = (2z + (1+2im)) / (3z)
            m = MobiusSphere.MobiusTransformation(2.0, 1.0+2.0im, 3.0, 0.0)
            rt = MobiusSphere.decompose_to_rigid(m)
            s2 = rt(s1)
            proj_s2 = MobiusSphere.StereographicProjection(s2)

            # Test north pole mapping
            np_s2 = np(s2)
            np_proj = proj_s2(np_s2)
            @test isinf(np_proj)
            @test m(complex(Inf)) ≈ 2/3

            # Test sphere properties
            @test norm(s2.center) > 0
            @test s2.center[3] > -1  # As per problem requirement

            # # Test your specific cases
            # z = complex(1.5, -0.5)
            # P = proj_s1(z)
            # @test norm(P - s1.center) ≈ 1 atol = SPHERE_TOL
            # P_rt = rt(P)

            # # Point should be on target sphere
            # @test norm(P_rt - s2.center) ≈ 1 atol=SPHERE_TOL

            # # Projected point should match Möbius transformation
            # z_rt = proj_s2(P_rt)
            # expected = m(z)
            # @test z_rt ≈ expected atol = NUM_TOL

            z = complex(1.5, -0.5)
            P = proj_s1(z)
            @test norm(P - s1.center) ≈ 1 atol = SPHERE_TOL
            P_rt = rt(P)
            @test norm(P_rt - s2.center) ≈ 1 atol=SPHERE_TOL
            @test proj_s2(P_rt) ≈ m(z) atol = NUM_TOL

            # Additional test: north pole projection
            z = complex(Inf)
            P_inf = proj_s1(z)
            @test norm(P_inf - s1.center) ≈ 1 atol = SPHERE_TOL
            P_inf_rt = rt(P_inf)
            @test norm(P_inf_rt - s2.center) ≈ 1 atol=SPHERE_TOL
            @test proj_s2(P_inf_rt) ≈ m(z) atol = NUM_TOL
        end

        @testset "Edge Cases" begin
            # Identity transformation
            m_id = MobiusSphere.MobiusTransformation(1.0, 0.0, 0.0, 1.0)
            rt_id = MobiusSphere.decompose_to_rigid(m_id)
            @test rt_id.rotation ≈ one(SMatrix{3,3}) atol=NUM_TOL
            @test rt_id.translation ≈ zeros(3) atol=NUM_TOL

            # Inversion
            m_inv = MobiusSphere.MobiusTransformation(0.0, 1.0, 1.0, 0.0)
            rt_inv = MobiusSphere.decompose_to_rigid(m_inv)
            p = rt_inv([0,0,1])
            q = rt_inv([0,0,0])
            @test p ≈ [0,0,-1] atol= SPHERE_TOL
            @test q ≈ [0,0,0] atol= SPHERE_TOL

            # Test degenerate case
            m_zero = MobiusSphere.MobiusTransformation(0.0, 0.0, 0.0, 0.0)
            @test_throws ArgumentError MobiusSphere.decompose_to_rigid(m_zero)
        end

        @testset "StereographicProjection Callable" begin
            # Test on non-standard sphere
            s2 = MobiusSphere.Sphere([1,0,0])
            proj_s2 = MobiusSphere.StereographicProjection(s2)

            # Test projection of south pole
            P_south = [1,0,-1]
            z = proj_s2(P_south)
            @test z ≈ complex(1,0) atol=NUM_TOL

            # Test inverse projection of complex point
            P_back = proj_s2(complex(1,0))
            @test norm(P_south - P_back) < NUM_TOL

            # Test north pole handling
            @test proj_s2(np(s2)) == complex(Inf)
            @test proj_s2(complex(Inf)) ≈ np(s2) atol=NUM_TOL

            # Round-trip test for arbitrary point
            P_test = [1+0.5, 0.2, -sqrt(1 - 0.5^2 - 0.2^2)]
            # On sphere: (x-1)^2 + y^2 + z^2 = 1
            z_test = proj_s2(P_test)
            P_test_back = proj_s2(z_test)
            @test norm(P_test - P_test_back) < NUM_TOL
        end
    end
end

println("All tests passed successfully!")
