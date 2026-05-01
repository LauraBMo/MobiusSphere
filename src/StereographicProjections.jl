# Stereographic projections
struct StereographicProjection{T}
    # sphere center coordinates
    x::T
    y::T
    z::T
end

stereo(v) = StereographicProjection{eltype(v)}(v...)
stereo_projection(v) = stereo(v)
stereo(::Type{T}=Int64) where T = stereo(zeros(T, 3))

# Helper function: get the center of the sphere we are projecting from.
vec(proj::StereographicProjection) = [proj.x, proj.y, proj.z]

# Helper function: compute north pole (focus of the projection)
const NorthAxis = 3 # y for POV-Ray
const OtherAxis = filter(x -> !(x == NorthAxis), 1:3)
const NorthDir = begin
    out = zeros(Int, 3)
    out[NorthAxis] = 1
    out
end

north_pole(proj::StereographicProjection) = vec(proj) + NorthDir

# Project sphere point to complex plane
function (proj::StereographicProjection)(P::AbstractVector{T}) where {T}
    np = north_pole(proj)
    dir = P - np

    # Handle north pole (no dir)
    if sum(abs2, dir) == 0
        return INF[]
    end

    # Compute parameter t using plane equation
    i = NorthAxis
    t = -np[i] * inv(dir[i])

    # Project to plane
    j, k = OtherAxis
    x_proj = np[j] + t * dir[j]
    y_proj = np[k] + t * dir[k]

    _complex(x_proj, y_proj)
end
# To deal with CalciumFieldElem, or Nemo shit in general.
_complex(x, y) = x + y * parent(x)(1*im)
_complex(x::Number, y::Number) = complex(x, y)

# So tempting... but type piracy.
# function Base.*(a::CalciumFieldElem, z::Complex{T}) where {T}
#     C = a.parent
#     x, y = reim(z)
#     return a*x + a*y*onei(C)
# end

# Project complex number to sphere
function (proj::StereographicProjection)(z)
    np = north_pole(proj)

    # Handle infinity
    if isinf(z)
        return np
    end

    # Unpack complex number
    x, y = reim(z)
    pz = [zero(x), zero(x), zero(x)]
    j, k = OtherAxis
    pz[j] = x
    pz[k] = y
    # P_plane = [x, y, zero(T)]
    # dir = P_plane - np

    # Direction vector
    # Do not need of type 'T'
    dir = pz - np

    # Solve quadratic equation for t
    ## b = 2*dot(dir, np - center)
    ## But, we have np-c = [0,0,1], so
    i = NorthAxis
    A = 2 * dir[i]
    B = sum(abs2, dir)

    # Use the non-zero solution
    t = -A * inv(B)

    # Calculate intersection point
    return np + t .* dir
end
