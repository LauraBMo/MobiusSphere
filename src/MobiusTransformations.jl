module MobiusTransformations

# import LinearAlgebra: det
using UnPack: @unpack

import Base: inv, show, hash, ==, *, ∘, isone, typeof

export Mobius, Möbius, stereo

include("StereographicProjections.jl")

# Tolerance constants
# const NUM_TOL = 1e-12 # for printing,...
# const SPHERE_TOL = 1e-8 # for detect north pole


# Infinite to completing complaxe plane.
const INF = Ref{Any}(complex(Inf))

"""
    set_infinity(infinity)

Sets the representation of infinity used by the package.
"""
function set_infinity(infinity)
    INF[] = infinity
end

infinity() = INF[]

# Möbius transformation
"""
    MobiusTransformation{T}

Represents a Möbius transformation of the form:
f(z) = (a*z + b) / (c*z + d)

# Fields
- `a::T, b::T, c::T, d::T`: Coefficients of the transformation
"""
struct MobiusTransformation{T} <: Function
    a::T
    b::T
    c::T
    d::T
end

# Dealing with diferent types
MobiusTransformation(a, b, c, d) = MobiusTransformation(promote(a, b, c, d)...)

# From array-alike objects
"""
    MobiusTransformation(A)

Creates a Möbius transformation from an array or tuple of coefficients [a, b, c, d].
"""
MobiusTransformation(A) = MobiusTransformation(A...)


# Computing Mobius transformations

# Möbius(...) calls:
# 4 args: Build MobiusTransformation (generic)
# 3 args: Target Number-like -> Standard transformation
# 6 args: Transformation source Number-like -> target Number-like
# 1 args: Target vector -> 3 args
# 2 args: Source, Target vectors -> 6 args
# 1 args: arg is a Type, returns Identity (default ComplexF64)
# ## moved to MobiusSphere ## 1 args: arg is a Matrix Q, returns "rotation by Q".

"""
    Möbius(a, b, c, d)

Creates a Möbius transformation with coefficients a, b, c, d.
"""
Möbius(a, b, c, d) = MobiusTransformation(a, b, c, d)

const Mobius = Möbius   # for us lazy Americans

"""
    Möbius(x, y, z)

Returns the Möbius transformation that maps `[0, 1, Inf]` to given points.
Values of `Inf` are permitted.
"""
function Möbius(x, y, z)
    if isinf(x)
        return Mobius(z, y - z, one(x), zero(x))
    elseif isinf(y)
        return Mobius(-z, x, -one(x), one(x))
    elseif isinf(z)
        return Mobius(y - x, x, zero(x), one(x))
    else
        xy, yz = y - x, z - y
        return Mobius(z * xy, x * yz, xy, yz)
    end
end

"""
    Möbius(x, y, z, X, Y, Z)

Returns the Möbius transformation that maps `[x, y, z]` to `[X, Y, Z]`.
Values of `Inf` are permitted.
"""
function Möbius(x, y, z, X, Y, Z)
    m_source = Möbius(x, y, z) # (0, 1, Inf) -> (x, y, z)
    m_image = Möbius(X, Y, Z) # (0, 1, Inf) -> (X, Y, Z)
    # No division performed.
    return m_image * inv(m_source) # (x, y, z) -> (X, Y, Z)
end

"""
    Möbius(target)

Returns the Möbius transformation that maps `[0, 1, Inf]` to given `target = [z1, z2, z3]`.
Values of `Inf` are permitted.
"""
Möbius(target) = Möbius(target...)


"""
    Möbius(source, target)

Returns the Möbius transformation that maps `source` to `target`.
Values of `Inf` are permitted.
"""
Möbius(source, target) = Möbius(source..., target...)

"""
    Möbius(::Type{T}=ComplexF64)

Returns the identity Möbius transformation of type `T`.
"""
Möbius(::Type{T}=Int64) where T = Möbius(one(T), zero(T), zero(T), one(T))

"""
    isone(m::MobiusTransformation)

Return `true` if `m` is the identity Mobius transformation and `false` otherwise.
"""
function isone(m::MobiusTransformation)
    @unpack a, b, c, d = m
    return iszero(b) && iszero(c) && (a == d)
end
==(m::MobiusTransformation, n::MobiusTransformation) = isone(m * inv(n))

# TODO
# ≈(m::MobiusTransformation, n::MobiusTransformation) = isapproxone(m*inv(n))
#

Base.eltype(_::MobiusTransformation{T}) where {T} = T

function Base.hash(m::MobiusTransformation, h::UInt64=UInt64(0))
    z = 0.0 + 0.0 * im # kludge to make -0.0 and -0.0im into +versions
    a = m(0) + z
    b = m(1) + z
    c = m(Inf) + z
    return hash(a, hash(b, hash(c, h)))
end

# Vectorized operations
Base.broadcastable(m::MobiusTransformation) = Ref(m)

function det(m::MobiusTransformation)
    @unpack a, b, c, d = m
    return a * d - b * c
end

"""
    normalize(m::MobiusTransformation)

Returns a Mobius transformation `m2` such that `m2 == m1` and `det(m2) = 1`.
"""
normalize(m::MobiusTransformation) = inv(det(m)) * m

function Base.Matrix(m::MobiusTransformation)
    @unpack a, b, c, d = m
    return [a b; c d]
end

# Inverse Möbius transformation
function Base.inv(m::MobiusTransformation)
    @unpack a, b, c, d = m
    MobiusTransformation(d, -b, -c, a)
end

# *(λ, m::MobiusTransformation) = MobiusTransformation(λ * Matrix(m))
# *(m::MobiusTransformation, λ) = *(λ, m)

"""
    *(m::MobiusTransformation, n::MobiusTransformation)

Compose two Möbius transformations.
"""
function *(m::MobiusTransformation, n::MobiusTransformation)
    @unpack a, b, c, d = n
    e, f, g, h = a, b, c, d
    @unpack a, b, c, d = m
    MobiusTransformation(a * e + b * g, a * f + b * h,
                         c * e + d * g, c * f + d * h)
end

"""
    ∘(m::MobiusTransformation, n::MobiusTransformation)

Compose two Möbius transformations (same as *).
"""
∘(m::MobiusTransformation, n::MobiusTransformation) = m * n

# Apply Möbius transformation
"""
    (m::MobiusTransformation)(z)

Apply to a complex number z the Möbius transformation
`m(z) = (a*z + b) / (c*z + d)`, where `m = Mobius([a b; c d])`.
"""
function (m::MobiusTransformation)(z)
    @unpack a, b, c, d = m
    if isinf(z)
        numer, denom = a, c
    else
        numer, denom = a * z + b, c * z + d
    end

    if abs(denom) == 0
        return INF[]
    else
        return numer * inv(denom)
    end
end

#
# Display
#

function show(io::IO, m::MobiusTransformation)
    @unpack a, b, c, d = m
    print(IOContext(io, :compact => true), "Möbius map z --> ($a*z + $b) / ($c*z + $d)")
end

function show(io::IO, ::MIME"text/plain", m::MobiusTransformation)
    string_linear((X, Y)) = "(" * X * ")*z + " * Y
    @unpack a, b, c, d = m
    # if abs(c) < NUM_TOL
    #     A, B = [repr("text/plain", x) for x in [a*inv(d), b*inv(d)]]
    #     numer = string_linear((A, B))
    #     print(io, "Möbius:\n   ", numer)
    # end
    A, B, C, D = [repr("text/plain", x) for x in [a, b, c, d]]
    numer, denom = string_linear.([(A, B), (C, D)])
    newline = "\n   "
    hline = reduce(*, fill("–", maximum(length, [numer, denom])))

    ## Print message
    print(io, "Möbius: ", eltype(m), newline,
        numer, newline,
        hline, newline,
        denom)
end

end # of module MobiusTransformations.
