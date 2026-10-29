#!/usr/bin/env python3
"""Audit corrected runs and export measured curves, tables, and TikZ figures.

Only Python's standard library is required. Run from the repository root:
python3 experiments/report.py --source results/horizon --output results/report
To update the manuscript, set --output to the manuscript directory.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("sbi", "pbi", "li", "heuristic")


def load_results(source, require_full=True):
    manifest = json.loads((source / "manifest.json").read_text())
    if require_full and manifest["mode"] != "full":
        raise ValueError("Publication export requires a full run, not pilot results")
    for name, expected in manifest["result_sha256"].items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Result hash mismatch: {name}")
    with (source / "runs.csv").open() as stream:
        runs = list(csv.DictReader(stream))
    with (source / "trajectories.csv").open() as stream:
        traces = list(csv.DictReader(stream))
    paired = {}
    for r in runs:
        horizon, fakes, target = (int(r[k]) for k in ("T", "fakes", "target"))
        genuine = [int(r[f"genuine{i}"]) for i in range(10)]
        injected = [int(r[f"fake{i}"]) for i in range(10)]
        means = [float(r[f"mu{i}"]) for i in range(10)]
        cost, ratio, lower = (float(r[k]) for k in ("cost", "ratio", "r_min"))
        if sum(genuine) + sum(injected) != horizon or sum(injected) != fakes:
            raise ValueError("Sample accounting mismatch")
        if injected[-1] or target != genuine[-1] or abs(ratio - target / horizon) > 1e-14:
            raise ValueError("Target-arm accounting mismatch")
        if not math.isfinite(cost) or cost < 0 or not 0 <= ratio <= 1:
            raise ValueError("Invalid cost or ratio")
        if r["method"] != "li":
            expected = sum(n * abs(lower - mean) for n, mean in zip(injected, means))
            if not math.isclose(cost, expected, rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("Cost is not the true-mean deviation cost")
        if "completed_arms" in r:
            if not 0 <= int(r["completed_arms"]) <= 9 or not 0 <= int(r["post_injection_pulls"]) <= sum(genuine[:-1]):
                raise ValueError("Invalid suppression diagnostics")
            if int(r["completed_arms"]) == 9 and r["method"] in ("sbi", "pbi", "li"):
                if any(int(r[f"prescribed{i}"]) != injected[i] for i in range(9)):
                    raise ValueError("Completed injection budgets must match actual counts")
        key = tuple(r[k] for k in ("learner", "grid", "x", "trial", "method"))
        if key in paired:
            raise ValueError("Duplicated run")
        paired[key] = r
    for key, r in paired.items():
        if r["method"] == "heuristic":
            sbi = paired[key[:-1] + ("sbi",)]
            if any(r[f"fake{i}"] != sbi[f"fake{i}"] for i in range(10)):
                raise ValueError("Heuristic budget is not matched to SBI")
            if not math.isclose(float(r["cost"]), float(sbi["cost"]), rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("Matched per-arm budgets must have matched true-mean costs")
    print(f"Audited {len(runs)} runs: sample totals, target counts, true-mean costs, and matched budgets")
    return runs, traces, manifest


def average(rows, column):
    return statistics.mean(float(r[column]) for r in rows)


def table(path, rows, fields):
    with path.open("w") as stream:
        stream.write(" ".join(fields) + "\n")
        for r in rows:
            stream.write(" ".join(format(r[k], ".17g") for k in fields) + "\n")


def series(rows, column, x_column="x"):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[float(r[x_column])][r["method"]].append(float(r[column]))
    result = []
    for x, methods in sorted(grouped.items()):
        row = {"x": x}
        for method in METHODS:
            values = methods[method]
            mean = statistics.mean(values)
            deviation = statistics.stdev(values) if len(values) > 1 else 0
            row.update({method: mean, method + "_low": mean - deviation, method + "_high": mean + deviation})
        result.append(row)
    return result


def figure(name, kind, values):
    peak = max(r[m + "_high"] for r in values for m in METHODS)
    maximum = max(r["x"] for r in values)
    options = [r"xlabel={Horizon $T$ ($10^6$)}",
               f"xmin=0, xmax={maximum:.17g}",
               r"xtick={0,250000,500000,750000,1000000}, xticklabels={0,0.25,0.5,0.75,1}, scaled x ticks=false"]
    if kind == "ratio":
        options.extend([r"title={Target-arm ratio}",
                        r"ymin=0, ymax=1.04, ytick={0,0.25,0.5,0.75,1}, scaled y ticks=false"])
    else:
        options.extend([r"title={Attack cost $C^F$}",
                        f"ymin=0, ymax={max(1, peak * 1.1):.10g}", r"scaled y ticks=true"])
    lines = ["% Generated from corrected experiments; markers are measured points.",
             r"\begin{tikzpicture}", r"\begin{axis}[fdi axis," + ",\n    ".join(options) + "]"]
    for method in METHODS:
        lines.append(r"\addplot[fdi " + method + " band, forget plot] table[x=x,y=" + method
                     + "] {fig/data/" + name + r"-bands.dat} \closedcycle;")
    for method in METHODS:
        style = "fdi " + method + (", densely dashed" if method == "heuristic" else "")
        lines.append(r"\addplot[" + style + ", mark=none] table[x=x,y=" + method + "] {fig/data/" + name + ".dat};")
        lines.append(r"\addplot[" + style + ", only marks, forget plot] table[x=x,y=" + method + "] {fig/data/" + name + "-marks.dat};")
    lines.extend([r"\end{axis}", r"\end{tikzpicture}", ""])
    return "\n".join(lines)


def summaries(runs, target):
    summaries = []
    grouped = defaultdict(list)
    for r in runs:
        grouped[(r["learner"], r["grid"], float(r["x"]), r["method"])].append(r)
    for (learner, grid, x, method), rows in sorted(grouped.items()):
        summaries.append({"learner": learner, "grid": grid, "x": x, "method": method,
                          "trials": len(rows), "mean_cost": average(rows, "cost"),
                          "mean_fakes": average(rows, "fakes"), "mean_ratio": average(rows, "ratio"),
                          "infeasible_trials": sum(int(r["infeasible_arms"]) > 0 for r in rows),
                          "ratio_at_least_95pct_trials": sum(float(r["ratio"]) >= 0.95 for r in rows),
                          "fully_suppressed_trials": sum(int(r.get("completed_arms", -1)) == 9 and
                              int(r.get("post_injection_pulls", -1)) == 0 and
                              int(r.get("post_first_injection_pulls", -1)) == 0 and int(r["infeasible_arms"]) == 0 for r in rows)})
    with (target / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summaries)
    (target / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    tex = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Learner & Method & $N^F$ & $C^F$ & Target (\%) & Infeasible trials \\", r"\midrule"]
    for learner in ("ucb", "ts"):
        endpoint = max(float(r["T"]) for r in runs if r["grid"] == "horizon" and r["learner"] == learner)
        for method in METHODS:
            s = next(s for s in summaries if s["learner"] == learner and s["grid"] == "horizon"
                     and s["x"] == endpoint and s["method"] == method)
            name = "Heuristic" if method == "heuristic" else method.upper()
            tex.append(f"{learner.upper()} & {name} & {s['mean_fakes']:.1f} & {s['mean_cost']:.1f} & "
                       + f"{100*s['mean_ratio']:.2f} & {s['infeasible_trials']}/{s['trials']}" + r" \\")
        if learner == "ucb": tex.append(r"\midrule")
    tex.extend([r"\bottomrule", r"\end{tabular}", ""])
    (target / "attack-summary.tex").write_text("\n".join(tex))
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Learner & Batch $f$ & $N^F$ & $C^F$ & Target (\%) & Infeasible trials \\", r"\midrule"]
    for learner in ("ucb", "ts"):
        for s in summaries:
            if s["learner"] == learner and s["grid"] == "batch" and s["method"] == "pbi":
                tex.append(f"{learner.upper()} & {int(s['x'])} & {s['mean_fakes']:.1f} & {s['mean_cost']:.1f} & "
                           + f"{100*s['mean_ratio']:.2f} & {s['infeasible_trials']}/{s['trials']}" + r" \\")
        if learner == "ucb": tex.append(r"\midrule")
    tex.extend([r"\bottomrule", r"\end{tabular}", ""])
    (target / "batch-feasibility.tex").write_text("\n".join(tex))
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Learner & Fake reward & $N^F$ & $C^F$ & Target (\%) & Infeasible trials \\", r"\midrule"]
    for learner in ("ucb", "ts"):
        selected = [s for s in summaries if s["learner"] == learner and s["grid"] == "strength" and s["method"] == "pbi"]
        for s in sorted(selected, key=lambda s: s["x"], reverse=True):
            tex.append(f"{learner.upper()} & {int(s['x'])} & {s['mean_fakes']:.1f} & {s['mean_cost']:.1f} & "
                       + f"{100*s['mean_ratio']:.2f} & {s['infeasible_trials']}/{s['trials']}" + r" \\")
        if learner == "ucb": tex.append(r"\midrule")
    tex.extend([r"\bottomrule", r"\end{tabular}", ""])
    (target / "reward-sensitivity.tex").write_text("\n".join(tex))


def export(source, output):
    runs, _, manifest = load_results(source)
    horizon_runs = [r for r in runs if r["grid"] == "horizon"]
    if not horizon_runs:
        raise ValueError("Publication figures require varying-horizon endpoint runs")
    horizons = {int(r["T"]) for r in horizon_runs}
    expected = {(learner, horizon, trial, method)
                for learner in ("ucb", "ts") for horizon in horizons
                for trial in range(20) for method in METHODS}
    actual = {(r["learner"], int(r["T"]), int(r["trial"]), r["method"]) for r in horizon_runs}
    if actual != expected or len(horizon_runs) != len(expected):
        raise ValueError("Both panels require the same complete grid of 20 trials and four methods")
    fig = output / "fig"; target = fig / "data"; target.mkdir(parents=True, exist_ok=True)
    # Parameter sweeps remain in raw results and summary tables, not the main figures.
    for learner in ("ucb", "ts"):
        name = learner + "-cost-delta"
        for suffix in (".tex", ".pdf"):
            path = fig / (name + suffix)
            if path.exists(): path.unlink()
        for path in target.glob(name + "*.dat"): path.unlink()
    fields = ["x"] + [k for m in METHODS for k in (m, m + "_low", m + "_high")]
    provenance = dict(manifest)
    provenance["plots"] = {}
    for learner in ("ucb", "ts"):
        for kind in ("horizon", "ratio"):
            name = learner + "-" + ("selection-ratio" if kind == "ratio" else "cost-" + kind)
            rows = [r for r in runs if r["learner"] == learner and r["grid"] == "horizon"]
            values = series(rows, "ratio" if kind == "ratio" else "cost", "T")
            table(target / (name + ".dat"), values, fields)
            # Every plotted marker is an independently simulated horizon endpoint.
            markers = values
            table(target / (name + "-marks.dat"), markers, fields)
            bands = [{"x": r["x"], **{m: r[m + "_low"] for m in METHODS}} for r in values]
            bands += [{"x": r["x"], **{m: r[m + "_high"] for m in METHODS}} for r in reversed(values)]
            table(target / (name + "-bands.dat"), bands, ["x", *METHODS])
            (fig / (name + ".tex")).write_text(figure(name, kind, values))
            trials = len({r["trial"] for r in rows})
            provenance["plots"][name] = {"rows": len(values), "trials": trials,
                "interpolation": "Straight segments between recorded points; no extrapolation.",
                "markers": "Every independently simulated horizon endpoint.",
                "source": "runs.csv: same learner, horizon, trial, and method for both panels",
                "error_band": "mean plus/minus one sample standard deviation (ddof=1)",
                "horizon_axis": "linear"}
    summaries(runs, target)
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported four corrected figures and three summary tables to {fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    local_clone = ROOT / "experiments/bandit-fake-data-injection/results/horizon"
    parser.add_argument("--source", type=Path, default=local_clone if local_clone.exists() else ROOT / "results/horizon")
    parser.add_argument("--output", type=Path, default=ROOT if (ROOT / "0-main.tex").exists() else ROOT / "results/report")
    parser.add_argument("--audit-only", action="store_true", help="Validate runs without exporting publication figures")
    arguments = parser.parse_args()
    if arguments.audit_only:
        load_results(arguments.source.resolve(), require_full=False)
    else:
        export(arguments.source.resolve(), arguments.output.resolve())
