using Documenter, DeepViz

makedocs(;
    modules=[DeepViz],
    format=Documenter.HTML(),
    pages=[
        "About" => "index.md",
        "Getting Started" => "gs.md",
        "Insights" => "insights.md",
        "Algorithms Reference" => "api.md",
        "Utility Function" => "utility.md",
        "Usage" => "usage.md",
        "FAQs" => "faq.md"
    ],
    repo="https://github.com/AdarshKumar712/DeepViz.jl/blob/{commit}{path}#L{line}",
    sitename="DeepViz.jl",
    authors="Adarshkumar712 <Adarshkumar712.ak@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/AdarshKumar712/DeepViz.jl",
)
