module MobiusTransformations

import Base: inv, show, hash, ==, *, ∘, isone, typeof

export Mobius, Möbius, stereo

include("StereographicProjections.jl")

const INF = Ref{Any}(complex(Inf))

"""
    set_infinity(infinity)

Sets the representation of infinity used by the package.
"""
function set_infinity(infinity)
    INF[] = infinity
end

infinity() = INF[]

"""
    MobiusTransformation{T}

Represents a Möbius transformation `f(z) = (a*z + b) / (c*z + d)`.
"""
struct MobiusTransformation{T} <: Function
    a::T
    b::T
    c::T
    d::T
end

MobiusTransformation(a, b, c, d) = MobiusTransformation(promote(a, b, c, d)...)
MobiusTransformation(A) = MobiusTransformation(A...)

"""
    Möbius(a, b, c, d)

Creates a Möbius transformation with coefficients a, b, c, d.
"""
Möbius(a, b, c, d) = MobiusTransformation(a, b, c, d)

const Mobius = Möbius

"""
    Möbius(x, y, z)

Returns the Möbius transformation that maps `[0, 1, Inf]` to the given points.
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

Returns the Möbius transformation mapping `[x, y, z]` to `[X, Y, Z]`.
"""
function Möbius(x, y, z, X, Y, Z)
    m_source = Möbius(x, y, z)
    m_image  = Möbius(X, Y, Z)
    return m_image * inv(m_source)
end

"""
    Möbius(target)

Returns the Möbius transformation mapping `[0, 1, Inf]` to `target = [z1, z2, z3]`.
"""
Möbius(target) = Möbius(target...)

"""
    Möbius(source, target)

Returns the Möbius transformation mapping `source` to `target`.
"""
Möbius(source, target) = Möbius(source..., target...)

"""
    Möbius(::Type{T}=Int64)

Returns the identity Möbius transformation of type `T`.
"""
Möbius(::Type{T}=Int64) where {T} = Möbius(one(T), zero(T), zero(T), one(T))

"""
    isone(m::MobiusTransformation)

Return `true` if `m` is the identity transformation.
"""
function isone(m::MobiusTransformation)
    (; a, b, c, d) = m
    return iszero(b) && iszero(c) && (a == d)
end

==(m::MobiusTransformation, n::MobiusTransformation) = isone(m * inv(n))

Base.eltype(_::MobiusTransformation{T}) where {T} = T

function Base.hash(m::MobiusTransformation, h::UInt64=UInt64(0))
    z = 0.0 + 0.0 * im
    a = m(0) + z
    b = m(1) + z
    c = m(Inf) + z
    return hash(a, hash(b, hash(c, h)))
end

Base.broadcastable(m::MobiusTransformation) = Ref(m)

function det(m::MobiusTransformation)
    (; a, b, c, d) = m
    return a * d - b * c
end

"""
    normalize(m::MobiusTransformation)

Returns a Möbius transformation `m2` such that `m2 == m` and `det(m2) = 1`.
"""
normalize(m::MobiusTransformation) = inv(det(m)) * m

function Base.Matrix(m::MobiusTransformation)
    (; a, b, c, d) = m
    return [a b; c d]
end

function Base.inv(m::MobiusTransformation)
    (; a, b, c, d) = m
    MobiusTransformation(d, -b, -c, a)
end

"""
    *(m::MobiusTransformation, n::MobiusTransformation)

Compose two Möbius transformations.
"""
function *(m::MobiusTransformation, n::MobiusTransformation)
    (; a, b, c, d) = n
    e, f, g, h = a, b, c, d
    (; a, b, c, d) = m
    MobiusTransformation(a * e + b * g, a * f + b * h,
                         c * e + d * g, c * f + d * h)
end

∘(m::MobiusTransformation, n::MobiusTransformation) = m * n

"""
    (m::MobiusTransformation)(z)

Apply `m(z) = (a*z + b) / (c*z + d)`.
"""
function (m::MobiusTransformation)(z)
    (; a, b, c, d) = m
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

function show(io::IO, m::MobiusTransformation)
    (; a, b, c, d) = m
    print(IOContext(io, :compact => true), "Möbius map z --> ($a*z + $b) / ($c*z + $d)")
end

function show(io::IO, ::MIME"text/plain", m::MobiusTransformation)
    string_linear((X, Y)) = "(" * X * ")*z + " * Y
    (; a, b, c, d) = m
    A, B, C, D = [repr("text/plain", x) for x in [a, b, c, d]]
    numer, denom = string_linear.([(A, B), (C, D)])
    newline = "\n   "
    hline = reduce(*, fill("–", maximum(length, [numer, denom])))
    print(io, "Möbius: ", eltype(m), newline,
        numer, newline,
        hline, newline,
        denom)
end

end # of module MobiusTransformations
