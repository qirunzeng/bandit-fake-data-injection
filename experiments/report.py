#!/usr/bin/env python3
"""Audit corrected runs and export measured curves, tables, and TikZ figures.

Only Python's standard library is required. Run from the repository root:
python3 experiments/report.py --source results/corrected --output results/report
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
    if kind == "ratio":
        options = [r"xlabel={Record index $t$ ($10^4$)}, title={Target-arm ratio}",
                   r"xmin=0, xmax=100000, xtick={0,25000,50000,75000,100000}",
                   r"xticklabels={0,2.5,5,7.5,10}, scaled x ticks=false",
                   r"ymin=0, ymax=1.04, ytick={0,0.25,0.5,0.75,1}, scaled y ticks=false"]
    else:
        if kind == "delta":
            options = [r"xlabel={$\delta_0$}, title={Attack cost $C^F$}",
                       r"xmin=0.1, xmax=0.5, xtick={0.1,0.3,0.5}"]
        else:
            options = [r"xlabel={Horizon $T$}, title={Attack cost $C^F$}",
                       r"xmode=log, xmin=10, xmax=10000000, xtick={10,1000,100000,10000000}"]
        options.extend([f"ymin=0, ymax={max(1, peak * 1.1):.10g}", r"scaled y ticks=true"])
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
                          "ratio_at_least_95pct_trials": sum(float(r["ratio"]) >= 0.95 for r in rows)})
    with (target / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (target / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    tex = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"Learner & Method & $N^F$ & $C^F$ & Target (\%) & Infeasible trials \\", r"\midrule"]
    for learner in ("ucb", "ts"):
        for method in ("sbi", "pbi", "li", "heuristic"):
            s = next(s for s in summaries if s["learner"] == learner and s["grid"] == "ratio" and s["method"] == method)
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
    runs, traces, manifest = load_results(source)
    fig = output / "fig"; target = fig / "data"; target.mkdir(parents=True, exist_ok=True)
    fields = ["x"] + [k for m in METHODS for k in (m, m + "_low", m + "_high")]
    provenance = dict(manifest)
    provenance["plots"] = {}
    for learner in ("ucb", "ts"):
        for kind in ("delta", "horizon", "ratio"):
            name = learner + "-" + ("selection-ratio" if kind == "ratio" else "cost-" + kind)
            rows = [r for r in (traces if kind == "ratio" else runs) if r["learner"] == learner
                    and (kind == "ratio" or r["grid"] == kind)]
            values = series(rows, "ratio" if kind == "ratio" else "cost", "t" if kind == "ratio" else "x")
            table(target / (name + ".dat"), values, fields)
            if kind == "ratio":
                positions = [5000 + 10000 * j for j in range(10)]
                markers = [min(values, key=lambda r: abs(r["x"] - x)) for x in positions]
            elif kind == "horizon":
                positions = [math.exp(math.log(values[0]["x"]) + j / 9 *
                             (math.log(values[-1]["x"]) - math.log(values[0]["x"]))) for j in range(10)]
                markers = [min(values, key=lambda r: abs(math.log(r["x"]) - math.log(x))) for x in positions]
            else:
                indices = sorted({round(j * (len(values) - 1) / 9) for j in range(10)})
                markers = [values[j] for j in indices]
            table(target / (name + "-marks.dat"), markers, fields)
            bands = [{"x": r["x"], **{m: r[m + "_low"] for m in METHODS}} for r in values]
            bands += [{"x": r["x"], **{m: r[m + "_high"] for m in METHODS}} for r in reversed(values)]
            table(target / (name + "-bands.dat"), bands, ["x", *METHODS])
            (fig / (name + ".tex")).write_text(figure(name, kind, values))
            trials = len({r["trial"] for r in rows})
            provenance["plots"][name] = {"rows": len(values), "trials": trials,
                "interpolation": "Straight segments between recorded points; no extrapolation.",
                "markers": "Recorded sweep points or recorded trajectory checkpoints.",
                "error_band": "mean plus/minus one sample standard deviation (ddof=1)",
                "horizon_axis": "logarithmic" if kind == "horizon" else "linear"}
    summaries(runs, target)
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported six corrected figures and three summary tables to {fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    local_clone = ROOT / "experiments/bandit-fake-data-injection/results/corrected"
    parser.add_argument("--source", type=Path, default=local_clone if local_clone.exists() else ROOT / "results/corrected")
    parser.add_argument("--output", type=Path, default=ROOT if (ROOT / "0-main.tex").exists() else ROOT / "results/report")
    parser.add_argument("--audit-only", action="store_true", help="Validate runs without exporting publication figures")
    arguments = parser.parse_args()
    if arguments.audit_only:
        load_results(arguments.source.resolve(), require_full=False)
    else:
        export(arguments.source.resolve(), arguments.output.resolve())
