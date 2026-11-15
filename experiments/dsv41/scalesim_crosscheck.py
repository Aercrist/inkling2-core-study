"""Cross-check the simulator's matrix-engine utilization against SCALE-Sim v3 (cycle-level
systolic-array simulator, github.com/scalesim-project/scale-sim-v3).

Question: what utilization does a matrix engine achieve on DeepSeek-V4.1-Flash decode GEMMs
(M = 6 verify rows, or ~1 row per routed expert) versus prefill GEMMs, for a conventional
128x128 array and for MDA-1's 16-row x wide-N engine? Our simulator models utilization as
mm_eff x M / (ceil(M / array_rows) * array_rows). SCALE-Sim gives cycle counts including
pipeline fill/drain. Skipped (with a note in results) if scalesim is not installed.
"""
from __future__ import annotations

import contextlib
import csv
import io
import math
import pathlib
import tempfile

from experiments.dsv41.common import RESULTS, save_json

SHAPES = [  # (name, M, N, K) per chip at TP4
    ("decode routed expert w13 (1 row/expert)", 1, 1152, 5120),
    ("decode wq_b shard (6 rows)", 6, 8192, 1280),
    ("decode batch-64 expert w13 (6 rows/expert)", 6, 1152, 5120),
    ("prefill expert w13 (8K chunk, 128 rows/expert)", 128, 1152, 5120),
]
ARRAYS = [("128x128 WS", 128, 128, "ws"), ("128x128 OS", 128, 128, "os"), ("16x1024 OS (MDA-1)", 16, 1024, "os")]

CFG = """[general]
run_name = {name}

[run_presets]
InterfaceBandwidth = CALC
UseRamulatorTrace = False

[architecture_presets]
ArrayHeight = {rows}
ArrayWidth = {cols}
IfmapSramSzkB = 8192
FilterSramSzkB = 8192
OfmapSramSzkB = 2048
IfmapOffset = 0
FilterOffset = 10000000
OfmapOffset = 20000000
Dataflow = {df}
ReadRequestBuffer = 32
WriteRequestBuffer = 32

[layout]
IfmapCustomLayout = False
FilterCustomLayout = False
IfmapSRAMBankBandwidth = 10
IfmapSRAMBankNum = 10
IfmapSRAMBankPort = 2
FilterSRAMBankBandwidth = 10
FilterSRAMBankNum = 10
FilterSRAMBankPort = 2

[sparsity]
SparsitySupport = false
SparseRep = ellpack_block
OptimizedMapping = false
BlockSize = 8
RandomNumberGeneratorSeed = 40
"""


from sim.hardware import load_hw  # noqa: E402

MM_EFF = load_hw("configs/hw/dsv41/asic_mda1.yaml").mm_eff


def analytic_util(M: int, rows: int, mm_eff: float = 0.8) -> float:
    return mm_eff * M / (math.ceil(M / rows) * rows)


def _compute_only(cfg_path: pathlib.Path, topo_path: pathlib.Path, lay_path: pathlib.Path):
    """Drive SCALE-Sim's operand-matrix and systolic-compute stages directly. The demand matrices
    have one row per array cycle (fill, steady state and drain included). The memory stage is
    skipped: v3.0.0's scratchpad model fails under numpy 2, and decode is HBM-bound anyway, which
    our own simulator handles."""
    from scalesim.compute.operand_matrix import operand_matrix
    from scalesim.compute.systolic_compute_is import systolic_compute_is
    from scalesim.compute.systolic_compute_os import systolic_compute_os
    from scalesim.compute.systolic_compute_ws import systolic_compute_ws
    from scalesim.layout_utils import layouts
    from scalesim.scale_config import scale_config
    from scalesim.topology_utils import topologies

    cfg = scale_config()
    cfg.read_conf_file(str(cfg_path))
    topo = topologies()
    topo.load_arrays(topofile=str(topo_path), mnk_inputs=True)
    lay = layouts()
    lay.load_arrays(layoutfile=str(lay_path))
    op = operand_matrix()
    op.set_params(config_obj=cfg, topoutil_obj=topo, layoututil_obj=lay, layer_id=0)
    op.create_operand_matrices()
    _, ifm = op.get_ifmap_matrix()
    _, flt = op.get_filter_matrix()
    _, ofm = op.get_ofmap_matrix()
    cs = {"os": systolic_compute_os, "ws": systolic_compute_ws, "is": systolic_compute_is}[cfg.get_dataflow()]()
    cs.set_params(config_obj=cfg, ifmap_op_mat=ifm, filter_op_mat=flt, ofmap_op_mat=ofm)
    cs.create_demand_matrices()
    _, _, ofd = cs.get_demand_matrices()
    return int(ofd.shape[0]), float(cs.get_avg_compute_utilization())


def run() -> dict:
    try:
        import scalesim  # noqa: F401
    except Exception as e:  # pragma: no cover
        out = {"skipped": f"scalesim not available: {e}"}
        save_json(out, RESULTS / "scalesim.json")
        return out
    rows_out = []
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        for an, R, C, df in ARRAYS:
            for sn, M, N, K in SHAPES:
                tag = "".join(ch if ch.isalnum() else "_" for ch in f"{an}_{sn}")
                cfg = td / f"{tag}.cfg"
                cfg.write_text(CFG.format(name=tag, rows=R, cols=C, df=df))
                topo = td / f"{tag}.csv"
                topo.write_text(f"Layer, M, N, K,\n{tag}, {M}, {N}, {K},\n")
                lay = td / f"{tag}_layout.csv"
                lay.write_text("Layer name," + ",".join(f"c{i}" for i in range(20)) + ",\n"
                               + tag + ",1,1,1,1,1,1,0,1,2,0,1,2,0,1,2,3,0,1,2,3,\n")
                with contextlib.redirect_stdout(io.StringIO()):
                    cycles, util_fold = _compute_only(cfg, topo, lay)
                ideal = M * N * K / (R * C)
                rows_out.append({"array": an, "shape": sn, "M": M, "N": N, "K": K, "cycles": cycles,
                                 "scalesim_util": ideal / cycles, "scalesim_mapping_util": util_fold,
                                 "model_util": analytic_util(M, R, MM_EFF) if (df == "os" and R == 16) else None})
    out = {"rows": rows_out, "tool": "SCALE-Sim v3.0.0 (pip scalesim), compute stage only",
           "note": "util = ideal MACs / (array MACs x cycles), including fill/drain. model_util is the simulator's "
                   "assumption for MDA-1's 16-row engine: mm_eff x M / (ceil(M/16) x 16). mm_eff was 0.8 before this "
                   "cross-check and was lowered to 0.5 because SCALE-Sim showed the 0.8 model 1.5-1.7x optimistic."}
    save_json(out, RESULTS / "scalesim.json")
    return out


if __name__ == "__main__":
    for r in run().get("rows", []):
        print(f'{r["array"]:20s} {r["shape"]:48s} cycles {r["cycles"]:>10.0f} util {r["scalesim_util"]:.3f} model {r["model_util"]}')
