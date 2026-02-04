module Visualization

using Dates
using Statistics: mean, std
using ..Backtesting: BacktestResult
using ..Optimization: OptimizationResult

# Core types
"""
    AbstractVisualization

Base type for visualization objects.
"""
abstract type AbstractVisualization end

"""
    VisualizationSpec

Lazy container for visualization configuration. Not rendered until displayed.
"""
struct VisualizationSpec{T}
    data::T
    view::Symbol
    options::Dict{Symbol,Any}
end

# Theme definitions - Professional styling
"""
    LIGHT_THEME

Default light theme configuration for visualizations.
"""
const LIGHT_THEME = Dict{Symbol,Any}(
    :backgroundcolor => "#fafafa",
    :textcolor => "#2d3748",
    :gridcolor => "#e2e8f0",
    :axiscolor => "#cbd5e0",
    :palette => ["#3182ce", "#e53e3e", "#38a169", "#805ad5", "#dd6b20", "#319795"],
    :fontsize => 12,
    :titlesize => 16,
    :ticksize => 10,
    :linewidth => 2.0,
    :spinewidth => 1.0,
)

"""
    DARK_THEME

Default dark theme configuration for visualizations.
"""
const DARK_THEME = Dict{Symbol,Any}(
    :backgroundcolor => "#1a1a2e",
    :textcolor => "#e2e8f0",
    :gridcolor => "#2d3748",
    :axiscolor => "#4a5568",
    :palette => ["#63b3ed", "#fc8181", "#68d391", "#b794f4", "#f6ad55", "#4fd1c5"],
    :fontsize => 12,
    :titlesize => 16,
    :ticksize => 10,
    :linewidth => 2.0,
    :spinewidth => 1.0,
)

# Semantic colors - Refined for both themes
"""
    COLORS

Semantic color palette used across visualizations.
"""
const COLORS = Dict{Symbol,String}(
    :profit => "#48bb78",
    :loss => "#f56565",
    :benchmark => "#a0aec0",
    :highlight => "#4299e1",
)

# Global state
const CURRENT_THEME = Ref{Dict{Symbol,Any}}(LIGHT_THEME)

"""
    set_theme!(theme::Symbol)

Set the global visualization theme. Options: `:light`, `:dark`.
"""
function set_theme!(theme::Symbol)
    if theme == :light
        CURRENT_THEME[] = LIGHT_THEME
    elseif theme == :dark
        CURRENT_THEME[] = DARK_THEME
    else
        error("Unknown theme: $theme. Use :light or :dark.")
    end
end

"""
    get_theme()

Get the current theme dictionary.
"""
get_theme() = CURRENT_THEME[]

"""
    visualize(data; kwargs...)
    visualize(data, view::Symbol; kwargs...)
    visualize(data, views::Vector{Symbol}; kwargs...)

Create a visualization specification for the given data.

# Arguments
- `data`: Data to visualize (BacktestResult, OptimizationResult, etc.)
- `view`: Specific view to render (e.g., `:equity`, `:drawdown`, `:frontier`)
- `views`: Multiple views to render as linked panels
- `theme`: Override theme (`:light`, `:dark`, or custom Dict)
- `backend`: Override backend (`:gl`, `:wgl`, `:cairo`)

# Examples
```julia
result = backtest(strategy, data)
visualize(result)                    # Default view
visualize(result, :drawdown)         # Specific view
visualize(result, [:equity, :drawdown])  # Multiple linked views
```
"""
function visualize end

# Default view dispatch
visualize(data; kwargs...) = visualize(data, default_view(data); kwargs...)

# Single view
function visualize(data, view::Symbol; theme=nothing, backend=nothing, kwargs...)
    opts = Dict{Symbol,Any}(kwargs...)
    if !isnothing(theme)
        opts[:theme] = theme isa Symbol ? (theme == :dark ? DARK_THEME : LIGHT_THEME) : theme
    end
    if !isnothing(backend)
        opts[:backend] = backend
    end
    VisualizationSpec(data, view, opts)
end

# Multiple views
function visualize(data, views::Vector{Symbol}; kwargs...)
    [visualize(data, v; kwargs...) for v in views]
end

# Default views for each type
default_view(::BacktestResult) = :dashboard
default_view(::OptimizationResult) = :frontier
default_view(::Any) = :auto

# Available views registry
const AVAILABLE_VIEWS = Dict{Type,Vector{Symbol}}(
    BacktestResult => [:dashboard, :equity, :drawdown, :returns, :rolling, :trades, :monthly, :tearsheet],
    OptimizationResult => [:frontier, :weights, :risk, :correlation],
)

"""
    available_views(data)

Return the list of available visualization views for the given data type.
"""
available_views(data) = get(AVAILABLE_VIEWS, typeof(data), Symbol[])

# ============================================================================
# LinkedContext for Interactive Plots
# ============================================================================

"""
    LinkedContext

Shared state for linked interactive plots. All plots sharing a context
will synchronize their cursors, zoom levels, and selections.
"""
mutable struct LinkedContext
    time_range::Tuple{Float64,Float64}
    cursor_time::Union{Float64,Nothing}
    selected_asset::Union{Symbol,Nothing}
    zoom_level::Float64

    function LinkedContext()
        new((0.0, 1.0), nothing, nothing, 1.0)
    end
end

# ============================================================================
# Render function - implemented by Makie extension
# ============================================================================

"""
    render(spec::VisualizationSpec)

Render a visualization specification to produce a plot.

This function is implemented by the Makie extension (QuantNovaMakieExt).
To use it, load Makie or one of its backends (CairoMakie, GLMakie, WGLMakie).

# Examples
```julia
using QuantNova
using CairoMakie  # or GLMakie, WGLMakie

result = backtest(strategy, data)
spec = visualize(result, :equity)
fig = render(spec)
```
"""
function render end

export AbstractVisualization, VisualizationSpec, LinkedContext
export visualize, set_theme!, get_theme, available_views
export LIGHT_THEME, DARK_THEME, COLORS
export render

# ============================================================================
# Dashboard Types
# ============================================================================

"""
    Row(items...; weight=1)

A row in a dashboard layout.
"""
struct Row
    items::Vector{Any}
    weight::Int

    Row(items...; weight=1) = new(collect(items), weight)
end

"""
    Dashboard

A multi-panel dashboard layout.

# Example
```julia
dashboard = Dashboard(
    title = "Strategy Monitor",
    theme = :dark,
    layout = [
        Row(visualize(result, :equity), weight=2),
        Row(visualize(result, :drawdown), visualize(result, :returns)),
    ]
)
serve(dashboard)
```
"""
struct Dashboard
    title::String
    theme::Symbol
    layout::Vector{Row}

    function Dashboard(; title="Dashboard", theme=:light, layout=Row[])
        new(title, theme, layout)
    end
end

"""
    serve(item; port=8080)

Serve a visualization or dashboard in the browser.
Requires WGLMakie and Bonito to be loaded.
"""
function serve end

"""
    save(filename::String, spec::VisualizationSpec; kwargs...)

Export a visualization to a file. Supported formats: PNG, PDF, SVG.

# Examples
```julia
save("report.png", visualize(result))
save("report.pdf", visualize(result); size=(800, 600))
```
"""
function save end

export Row, Dashboard, serve, save

# ============================================================================
# Precompilation
# ============================================================================

using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # Precompile theme operations
        set_theme!(:dark)
        set_theme!(:light)
        get_theme()

        # Precompile spec creation (doesn't require Makie)
        # These would be precompiled when BacktestResult is available
    end
end

end # module
