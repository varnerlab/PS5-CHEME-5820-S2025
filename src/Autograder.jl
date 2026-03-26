# Autograder.jl — lightweight autograder for PS5: Autoencoders
#
# Rubric scale (manual discussion review is separate)
#   0 = nothing runs
#   1 = errors, but at least one test passes
#   2 = errors, but most tests pass (at least one error remains)
#   3 = all tests pass, code runs
#   4 = all tests pass/code runs + manual discussion review is good

mutable struct Grader
    results :: Vector{NamedTuple{(:problem,:description,:earned,:total),
                                  Tuple{String,String,Int,Int}}}
end

Grader() = Grader(NamedTuple{(:problem,:description,:earned,:total),
                               Tuple{String,String,Int,Int}}[])

"""
    check!(grader, problem, description, pts, testfn)

Run `testfn()` and award `pts` if it returns `true`.  Re-running a check cell
after fixing code *updates* the existing entry rather than duplicating it, so
the score always reflects the most recent run.
"""
function check!(g::Grader, problem::String, description::String,
                pts::Int, testfn::Function)
    idx = findfirst(r -> r.problem == problem &&
                         r.description == description, g.results)
    try
        passed = (testfn() === true)
        earned = passed ? pts : 0
        entry  = (problem=problem, description=description, earned=earned, total=pts)
        isnothing(idx) ? push!(g.results, entry) : (g.results[idx] = entry)
        sym = passed ? "✓" : "✗"
        @printf("  %s  %2d / %2d pts  %s\n", sym, earned, pts, description)
    catch e
        entry = (problem=problem, description=description, earned=0, total=pts)
        isnothing(idx) ? push!(g.results, entry) : (g.results[idx] = entry)
        @printf("  ✗   0 / %2d pts  %s  [ERROR: %s]\n", pts, description, typeof(e))
    end
end

"""
    score!(grader)

Print a summary table of earned vs total points, grouped by problem.
"""
function score!(g::Grader; discussion_ok::Bool=false)
    isempty(g.results) && (println("No checks run yet."); return)
    e_total  = sum(r.earned for r in g.results)
    t_total  = sum(r.total  for r in g.results)
    n_pass   = count(r -> r.earned == r.total, g.results)
    n_total  = length(g.results)
    n_fail   = n_total - n_pass
    bar_w    = 12

    println()
    println("═"^56)
    @printf("  %-22s  %8s  %s\n", "Problem", "Score", "Progress")
    println("─"^56)
    for p in unique(r.problem for r in g.results)
        rows   = filter(r -> r.problem == p, g.results)
        pe     = sum(r.earned for r in rows)
        pt     = sum(r.total  for r in rows)
        filled = round(Int, (pe / max(pt, 1)) * bar_w)
        bar    = "█"^filled * "░"^(bar_w - filled)
        @printf("  %-22s  %3d / %3d  [%s]\n", p, pe, pt, bar)
    end
    println("─"^56)
    @printf("  %-22s  %3d / %3d\n", "AUTO-GRADED TOTAL", e_total, t_total)
    println("═"^56)

    rubric_score = if n_total == 0 || n_pass == 0
        0
    elseif n_fail > 0 && n_pass >= 1
        # Distinguish 1 vs 2 by whether most tests pass.
        n_pass > (n_total ÷ 2) ? 2 : 1
    else
        discussion_ok ? 4 : 3
    end

    @printf("  RUBRIC SCORE: %d / 4\n", rubric_score)
    @printf("  Tests passed: %d / %d\n", n_pass, n_total)
    @printf("  Discussion review status: %s\n", discussion_ok ? "approved" : "pending/manual")
    println("  Note: discussion answers are reviewed manually by instructors.")
    println("═"^56)
end
