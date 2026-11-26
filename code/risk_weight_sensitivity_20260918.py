"""Audit local weight sensitivity against the exact pixel-aggregated segment scores."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(r"D:\project\shuozhouinflucefactor")
source = root / "outputs" / "label_free_weight_search_20260909"
weights = pd.read_csv(source / "candidate_comparison.csv")
scores = pd.read_csv(source / "all_segment_scores.csv")
base = weights.loc[
    np.isclose(weights.w_environment, .10)
    & np.isclose(weights.w_insar, .70)
    & np.isclose(weights.w_landslide, .20)
].iloc[0]
base_id = int(base.candidate)
wcols = ["w_environment", "w_insar", "w_landslide"]
scenarios = weights.loc[
    (weights[wcols].sub(base[wcols].to_numpy(), axis=1).abs().sum(axis=1) <= .2000001)
    & (weights.candidate != base_id)
].copy()
scenarios = scenarios.loc[
    scenarios[wcols].sub(base[wcols].to_numpy(), axis=1).abs().max(axis=1) <= .1000001
]
result = []
for zone in ("protection", "control"):
    data = scores[scores.zone.eq(zone)].pivot(index="start_m", columns="candidate", values="risk_score").sort_index()
    b = data[base_id]
    top_n = int(np.ceil(.2 * len(b)))
    top_base = set(b.nlargest(top_n, keep="first").index)
    peak_base = float(b.idxmax())
    for _, scenario in scenarios.iterrows():
        candidate = int(scenario.candidate)
        a = data[candidate]
        top = set(a.nlargest(top_n, keep="first").index)
        result.append({
            "zone": zone,
            "candidate": candidate,
            "w_environment": float(scenario.w_environment),
            "w_insar": float(scenario.w_insar),
            "w_landslide": float(scenario.w_landslide),
            "spearman_rho": float(np.corrcoef(b.rank().to_numpy(), a.rank().to_numpy())[0, 1]),
            "top20_overlap_count": len(top_base & top),
            "top20_count": top_n,
            "top20_overlap": len(top_base & top) / top_n,
            "peak_start_m": float(a.idxmax()),
            "baseline_peak_start_m": peak_base,
            "baseline_peak_is_top20": peak_base in top,
            "750m_is_top20": 750.0 in top,
            "1500m_is_top20": 1500.0 in top,
        })

out = {
    "source": str(source / "all_segment_scores.csv"),
    "baseline_weights": [float(base[c]) for c in wcols],
    "baseline_candidate": base_id,
    "segment_count_per_zone": 156,
    "top20_segment_count": 32,
    "scenario_count": int(len(scenarios)),
    "results": result,
}
target = root / "outputs" / "risk_weight_sensitivity_20260918"
target.mkdir(exist_ok=True)
(target / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("scenarios", len(scenarios))
for zone in ("protection", "control"):
    sub = [r for r in result if r["zone"] == zone]
    print(zone, "rho", min(r["spearman_rho"] for r in sub), max(r["spearman_rho"] for r in sub),
          "overlap", min(r["top20_overlap"] for r in sub), max(r["top20_overlap"] for r in sub),
          "peaks", sorted(set(r["peak_start_m"] for r in sub)))
print("details", target / "results.json")
