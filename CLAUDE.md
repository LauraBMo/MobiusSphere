# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

**MobiusSphere** is a Julia package (v1.0.0-DEV) that implements bidirectional conversion between Möbius transformations (conformal maps on the Riemann sphere / complex plane) and rigid motions (rotation + translation) in 3D space. It uses exact arithmetic via Nemo/CalciumField in addition to standard floating-point.

## Commands

Julia is at `~/src/juliaup/bin/julia`. Always run from the package root with `--project=.`.

```bash
# Run full test suite
~/src/juliaup/bin/julia --project=. -e "using Pkg; Pkg.test()"

# Run tests directly (faster, same output)
~/src/juliaup/bin/julia --project=. test/runtests.jl

# Start a REPL with the package loaded
~/src/juliaup/bin/julia --project=. -e "using MobiusSphere"

# Add/update dependencies
~/src/juliaup/bin/julia --project=. -e "using Pkg; Pkg.add(\"PackageName\")"
```

## Architecture

### Module structure

`src/MobiusSphere.jl` is the top-level module. It `include`s the other source files and exports `Mobius_to_rigid!`.

| File | Responsibility |
|------|---------------|
| `src/MobiusTransformations.jl` | `MobiusTransformation{T}` struct — represents `f(z) = (az+b)/(cz+d)`; construction, composition, inversion, evaluation |
| `src/StereographicProjections.jl` | `StereographicProjection{T}` struct — maps between sphere points and complex plane; handles the north pole / infinity |
| `src/BaseMotions.jl` | Primitive sphere motions used by the decomposition algorithm: `Btonorth`, `Rtozero`, `Gtoone_step1`, `Gtoone_step2` |
| `src/MobiusSphere.jl` | `Mobius_to_rigid!` / `Mobius_to_rigid` / `rigid_to_Mobius` — core algorithm tying everything together |

### Core algorithm (`Mobius_to_rigid`)

The decomposition of a Möbius transformation into a rigid motion works by tracking three base points (R=0, G=1, B=∞ on the Riemann sphere) through four sequential primitive steps:

1. **`Btonorth(B)`** — rotate B to the north pole (yields rotation matrix)
2. **`Rtozero(zr)`** — translate R to the origin (horizontal translation)
3. **`Gtoone_step1(B, G)`** — translate G toward the unit circle
4. **`Gtoone_step2(zg)`** — rotate G to the point 1 on the unit circle

Each step returns a rigid motion component; they are composed to give the final rigid transformation. The inverse direction (`rigid_to_Mobius`) reconstructs the Möbius transformation from the rigid motion by applying the stereographic projection.

### Type generality

All structs are parameterized by `T` (the number field). The package is tested with both `Float64` and Nemo's `CalciumField` (exact algebraic arithmetic). The `__normalize` helper in `MobiusSphere.jl` dispatches on type to apply either Julia's `normalize` or Nemo's `canonical_unit`-based normalization.

### Naming

Both `Möbius` (Unicode) and `Mobius` (ASCII) are exported as aliases for `MobiusTransformation`.

## Dependencies

- **Nemo** — computer algebra system providing exact number fields (`CalciumField`, etc.)
- **NemoUtils** — utility helpers on top of Nemo
- **LinearAlgebra** — stdlib, used for matrix operations in rigid motions
