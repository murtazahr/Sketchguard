"""Byzantine-fraction sweep on Murmura's ZeroMQ distributed backend: measured bytes-on-wire for
BALANCE (full exchange) vs SketchGuard (two-phase sketch + selective fetch). Writes comm_sweep.json."""
import json, time, tempfile, os
from pathlib import Path
import numpy as np
from murmura.distributed.runner import DistributedRunner

N, D_HIDDEN, INPUT, CLASSES = 10, 1024, 100, 10
SKETCH, ROUNDS, RDUR = 1000, 4, 6.0
BYZ = [0.0, 0.1, 0.2, 0.3, 0.4]

def cfg(mode, byz, port_base):
    return f"""
experiment: {{name: "sweep-{mode}-{int(byz*100)}", seed: 42, rounds: {ROUNDS}, verbose: false}}
backend: distributed
distributed: {{round_duration_s: {RDUR}, base_port: {port_base}}}
topology: {{type: "fully", num_nodes: {N}, seed: 123}}
aggregation: {{algorithm: "{mode}", params: {{sketch_size: {SKETCH}, gamma: 2.0, kappa: 1.0, alpha: 0.5}}}}
attack: {{enabled: {str(byz>0).lower()}, type: "gaussian", percentage: {byz}, params: {{noise_std: 100.0}}}}
training: {{local_epochs: 1, batch_size: 32, lr: 0.01}}
data: {{adapter: "murmura.examples.synthetic.synthetic_adapter",
        params: {{num_nodes: {N}, samples_per_node: 48, input_dim: {INPUT}, num_classes: {CLASSES}, seed: 42}}}}
model: {{factory: "murmura.examples.synthetic.synthetic_mlp",
         params: {{input_dim: {INPUT}, hidden: {D_HIDDEN}, num_classes: {CLASSES}}}}}
"""

def run(mode, byz, port_base):
    p = Path(tempfile.gettempdir()) / f"cfg_{mode}_{int(byz*100)}.yaml"
    p.write_text(cfg(mode, byz, port_base))
    h = DistributedRunner(p).run(verbose=False)
    tx = h["comm_tx_bytes"]; ss = tx[1:] or tx
    return float(np.mean(ss)), float(np.mean((h["comm_tx_sketch"][1:] or h["comm_tx_sketch"])))

if __name__ == "__main__":
    d = int(100*INPUT*0 + INPUT*D_HIDDEN + D_HIDDEN + D_HIDDEN*CLASSES + CLASSES)
    rows = []
    port = 6000
    for bi, bz in enumerate(BYZ):
        bal, _ = run("balance", bz, port + bi*400); time.sleep(1)
        sg, sk = run("sketchguard", bz, port + bi*400 + 200); time.sleep(1)
        row = dict(byz=bz, balance_tx=bal, sketchguard_tx=sg, sketch_tx=sk,
                   saving_pct=100*(1-sg/bal), model_dim=d, sketch_size=SKETCH, n=N)
        rows.append(row)
        print(f"byz={bz:.2f}: BALANCE {bal/1e6:.2f}MB  SG {sg/1e6:.2f}MB  saving {row['saving_pct']:.1f}%", flush=True)
    out = Path(__file__).parent / "comm_sweep.json"
    out.write_text(json.dumps(rows, indent=2))
    print("wrote", out)
