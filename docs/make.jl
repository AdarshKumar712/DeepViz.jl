using Documenter, DeepViz

makedocs(;
    modules=[DeepViz],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/AdarshKumar712/DeepViz.jl/blob/{commit}{path}#L{line}",
    sitename="DeepViz.jl",
    authors="Adarshkumar712 <Adarshkumar712.ak@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/AdarshKumar712/DeepViz.jl",
)
