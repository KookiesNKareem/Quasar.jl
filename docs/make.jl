using Quasar
using Documenter

DocMeta.setdocmeta!(Quasar, :DocTestSetup, :(using Quasar); recursive=true)

makedocs(;
    modules=[Quasar],
    authors="Kareem Fareed",
    sitename="Quasar.jl",
    format=Documenter.HTML(;
        canonical="https://KookiesNKareem.github.io/Quasar.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "AD Backends" => "backends.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/KookiesNKareem/Quasar.jl",
    devbranch="main",
)
