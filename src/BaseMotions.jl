
I(x) = [one(x) zero(x) zero(x);
        zero(x) one(x) zero(x);
        zero(x) zero(x) one(x)]

Z(x) = [zero(x), zero(x), zero(x)]

# _set(A) = real.(calcium_to_complex.(A))

@inline _has_iszero(x) = Base.hasmethod(iszero, Tuple{typeof(x)})

function _approx_zero(x)
    if _has_iszero(x) && iszero(x)
        return true
    end

    ax = abs(x)
    if _has_iszero(ax) && iszero(ax)
        return true
    end

    if ax isa AbstractFloat
        return ax <= sqrt(eps(ax))
    end

    return false
end

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
    denom = one(z) + z
    if _approx_zero(denom)
        axis = [one(x) - x * x, -x * y, -x * z]
        axis_norm = sqrt(sum(abs2, axis))
        if _approx_zero(axis_norm)
            axis = [-x * y, one(y) - y * y, -y * z]
            axis_norm = sqrt(sum(abs2, axis))
        end
        if _approx_zero(axis_norm)
            throw(DomainError(B, "unable to determine rotation axis for south-pole input"))
        end
        axis_unit = axis ./ axis_norm
        rot = 2 .* (axis_unit * axis_unit') .- I(x)
    else
        rot = I(x) + M + inv(denom) .* (M^2)
    end
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
    d = __normalize(_norm(_det(x, z, a, c), _det(y, z, b, c)))
    if _approx_zero(d)
        return Z(x)
    end
    λ = -1 + (c - z) * inv(d)
    # tr = CT.AffineMap(I(x), λ .* B)
    tr = λ .* B
    return tr
end

# Step 4: Rotation by -angle(proj(G))
# We rotate proj(G)∈S^1, to 1.
function Gtoone_step2(zg)
    x, y = reim(zg)
    # θ = -atan(y // x)
    # M = [cos(θ) -sin(θ) 0;
    #      sin(θ)  cos(θ) 0;
    #      0       0      1]
    d = __normalize(_norm(x, y))
    if _approx_zero(d)
        return I(x)
    end
    pre_rot = inv(d) .* [x y;
                         -y x]
    rot = I(x)
    rot[1:2, 1:2] = pre_rot
    # @show rot = _set(rot)
    return rot
end
