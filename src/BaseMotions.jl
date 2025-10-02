
I(x) = [one(x) zero(x) zero(x);
        zero(x) one(x) zero(x);
        zero(x) zero(x) one(x)]

Z(x) = [zero(x), zero(x), zero(x)]

_set(A) = real.(calcium_to_complex.(A))
# Step 1: Rotate B to north pole.
# axis_rot1 = [-B[2],B[1],0] # = cross(B_rot, SVector(0, 0, 1))
# angle_rot1 = acos(B[3]) # = angle(B_rot, SVector(0, 0, 1))
function Btonorth(B)
    x, y, z = B
    M = fill(zero(x), 3, 3)
    M[1, 3] = -x
    M[2, 3] = -y
    M[3, 1] = x
    M[3, 2] = y
    # @show _set(M)
    # rot = CT.AffineMap(I(x) + M + inv(1 + z) .* (M^2), Z(x))
    rot = I(x) + M + inv(1 + z) .* (M^2)
    # @show _set(rot)
    # @show _set(rot*B)
    return rot
end

# Step 2: Horizontal translation by -proj(B)
function Rtozero(zr)
    x, y = reim(zr)
    # tr = CT.AffineMap(I(x), [-x, -y, zero(x)])
    tr = [-x, -y, zero(x)]
    return tr
end

_det(a, b, c, d) = a * d - b * c
_norm(a, b) = sqrt(a^2 + b^2)
# Step 3: Translation by λ(OB) (so B-north and R-zero remains, since ORB algined),
# with a λ such that ||proj_{B+λB}(G+λB)|| = 1
# So, we get B-north, R-zero, G-inS1 (module one).
function Gtoone_step1(B, G)
    a, b, c = B
    x, y, z = G
    d = complex_normal_form(_norm(_det(x, z, a, c), _det(y, z, b, c)))
    λ = -1 + (c - z) * inv(d)
    # tr = CT.AffineMap(I(x), λ .* B)
    tr = λ .* B
    return tr
end

# Step 4: Rotation by -angle(proj(G))
# We rotate proj(G)∈ S^1, to 1.
function Gtoone_step2(zg)
    x, y = reim(zg)
    # θ = -atan(y // x)
    # M = [cos(θ) -sin(θ) 0;
    #      sin(θ)  cos(θ) 0;
    #      0       0      1]
    d = complex_normal_form(_norm(x, y))
    M = [x y 0;
         -y x 0;
         0 0 0]
    rot = inv(d).*M
    M[3, 3] = 1
    return rot
end
