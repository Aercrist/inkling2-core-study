"""Calibrate the GPU model (G0) to the published Inkling 2 Small speeds; write results/calibration_gpu.json."""
from __future__ import annotations

import yaml

from experiments.common import FIG, PALETTE, RESULTS, ROOT, plt, save_json, style
import pathlib

from sim.calibrate_gpu import X0, Point, attention_kernel_check, fit, leave_one_out, load_points, predict, sensitivity
from sim.models.inkling2 import InklingShape


def main() -> dict:
    shape = InklingShape.from_checkpoint()
    pts, checks = load_points()
    pub = {m["id"]: m for m in yaml.safe_load((ROOT / "data" / "public_measurements.yaml").read_text())["measurements"]}
    prior = {p.id: predict(shape, p, X0)["value"] for p in pts}
    f = fit(shape, pts)
    x = f["x"]
    rows = []
    for p in pts:
        r = predict(shape, p, x)
        rows.append({"id": p.id, "src": pub[p.id]["src"], "hardware": pub[p.id]["hardware"], "engine": pub[p.id]["engine"],
                     "batch": p.batch, "context": p.context, "speculation": p.speculation, "accept_len": p.accept_len,
                     "prefill_tokens": p.prefill_tokens, "unit": "ms per forward" if p.prefill_tokens else "tok/s per user",
                     "published": p.published, "fitted": r["value"], "error": r["value"] / p.published - 1.0,
                     "prior_prediction": prior[p.id], "prior_error": prior[p.id] / p.published - 1.0,
                     "step_ms": 1e3 * r["step_time"], "prefill_share_of_gpu_time": r.get("prefill_share", 0.0),
                     "share": {k: r[k] / r["step_time"] for k in ("overhead", "memory", "compute", "collective")}})
    loo = leave_one_out(shape, pts, x)
    sens = sensitivity(shape, pts, x)
    # held-out checks: published figures whose in-flight batch is not published
    held = []
    for c in checks:
        lo, hi = c["batch_range"]
        preds = {}
        for b in sorted({lo, 2, 4, hi}):
            p = Point(id=c["id"], hw=c["hw"], n_gpus=c["n_gpus"], batch=b, context=c["context"], speculation=c["speculation"],
                      kv_format=c["kv_format"], target=c["target"], accept_len=c.get("accept_len", 3.5))
            preds[b] = predict(shape, p, x)["tok_s_user"]
        held.append({"id": c["id"], "src": pub[c["id"]]["src"], "published": c["target"], "predicted_by_in_flight_batch": preds,
                     "inside_range": min(preds.values()) <= c["target"] <= max(preds.values())})
    errs = sorted(abs(r["error"]) for r in loo)
    kc = yaml.safe_load((ROOT / "data" / "calibration_points.yaml").read_text())["kernel_checks"]
    kernel = [{"id": c["id"], "src": pub[c["id"]]["src"], "rows": attention_kernel_check(shape, c, x)} for c in kc]
    out = {
        "label": f"calibrated: 4 parameters fitted to {len(rows)} published figures",
        "params": f["params"], "prior_params": dict(zip(("hbm_eff", "kernel_overhead", "attn_eff", "row_overhead"), X0.tolist())),
        "dsv41_params": {"hbm_eff": 0.837504, "kernel_overhead": 4.214883e-6},
        "rms_log_error": f["rms_log_error"], "at_bound": f["at_bound"],
        "points": rows, "n_points": len(rows), "n_params": 4,
        "prior_abs_error_range": [min(abs(r["prior_error"]) for r in rows[:3]), max(abs(r["prior_error"]) for r in rows[:3])],
        "loo": loo, "loo_median_abs_error": errs[len(errs) // 2], "loo_max_abs_error": errs[-1],
        "sensitivity": sens,
        "param_ranges": {k: [min(s["params"][k] for s in sens), max(s["params"][k] for s in sens)] for k in f["params"]},
        "held_out": held, "kernel_checks": kernel,
    }
    save_json(out, RESULTS / "calibration_gpu.json")

    style()
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    lab = ["TokenSpeed 4xB300\nno speculation", "TokenSpeed 4xB300\nDSpark", "vLLM 4xGB200\nDSpark",
           "vLLM 8xGB200\n84 users, 32K", "vLLM 8xGB200\n72 users, 64K", "vLLM 8xGB200\n55 users, 128K", "vLLM 8xGB200\n37 users, 256K",
           "TokenSpeed 4xB300\nprefill 2K (ms)", "TokenSpeed 4xB300\nprefill 3K (ms)"]
    xs = range(len(rows))
    ax[0].bar([i - 0.2 for i in xs], [r["published"] for r in rows], 0.4, color="#bdc3c7", label="published (measured)")
    ax[0].bar([i + 0.2 for i in xs], [r["fitted"] for r in rows], 0.4, color=PALETTE["G0"], label="model, fitted")
    ax[0].set_xticks(list(xs)); ax[0].set_xticklabels(lab, rotation=60, ha="right", fontsize=6.5)
    ax[0].set_ylabel("tok/s per user (ms for prefill)"); ax[0].set_title("GPU model against published figures"); ax[0].legend(fontsize=7)
    ax[1].bar(list(xs), [100 * r["error"] for r in loo], color=PALETTE["G0"])
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(lab, rotation=60, ha="right", fontsize=6.5)
    ax[1].set_ylabel("error when the point is held out (%)"); ax[1].set_title("Leave-one-out (modeled)")
    fig.savefig(FIG / "fig_calibration_gpu.png"); plt.close(fig)
    return out


if __name__ == "__main__":
    o = main()
    print("params:", o["params"])
    for r in o["points"]:
        print(f"  {r['id']:34s} published {r['published']:6.1f} fitted {r['fitted']:7.1f} ({r['error']:+.1%}) prior {r['prior_prediction']:7.1f} ({r['prior_error']:+.1%})")
    for k in o["kernel_checks"]:
        print("  kernel check", [(r["context"], r["published_us"], round(r["model_us"], 1), f"{r['error']:+.0%}") for r in k["rows"]])
    print("LOO:", [(r["id"].split("_", 1)[1][:22], f"{r['error']:+.1%}") for r in o["loo"]], "median |err|", f"{o['loo_median_abs_error']:.1%}", "max", f"{o['loo_max_abs_error']:.1%}")
    for s in o["sensitivity"]:
        p = s["params"]
        print(f"  {s['alternative']:55s} eta {p['hbm_eff']:.3f} kappa {1e6 * p['kernel_overhead']:5.2f}us attn {p['attn_eff']:.4f} row {1e6 * p['row_overhead']:.3f}us rms {s['rms_log_error']:.3f} n={s['n_points']} bound={[k for k, v in s['at_bound'].items() if v]}")
    for h in o["held_out"]:
        print("  held out", h["id"], "published", h["published"], "predicted", {k: round(v) for k, v in h["predicted_by_in_flight_batch"].items()}, "inside:", h["inside_range"])
