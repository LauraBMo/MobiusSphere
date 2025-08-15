using MobiusSphere
using Documenter

DocMeta.setdocmeta!(MobiusSphere, :DocTestSetup, :(using MobiusSphere); recursive=true)

makedocs(;
    modules=[MobiusSphere],
    authors="LauBMo <laurea987@gmail.com> and contributors",
    sitename="MobiusSphere.jl",
    format=Documenter.HTML(;
        canonical="https://LauraBMo.github.io/MobiusSphere.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/LauraBMo/MobiusSphere.jl",
    devbranch="main",
)
