from pathlib import Path
from murmura.distributed.runner import DistributedRunner

def run(cfg):
    h = DistributedRunner(Path(cfg)).run(verbose=False)
    # steady-state mean tx bytes/round (skip round 1 warmup if >1 round)
    tx = h["comm_tx_bytes"]; sk = h["comm_tx_sketch"]; fu = h["comm_tx_full"]
    ss = tx[1:] or tx
    return sum(ss)/len(ss), tx, sk, fu

if __name__ == "__main__":
    bal, btx, _, bfu = run("experiments/configs/smoke_bal_distributed.yaml")
    sg,  stx, ssk, sfu = run("experiments/configs/smoke_sg_distributed.yaml")
    print("\n================ MEASURED COMMUNICATION (bytes/round, summed over nodes) ================")
    print(f"BALANCE (full exchange):   tx/round = {bal/1e6:.2f} MB   per-round {[round(x/1e6,2) for x in btx]}")
    print(f"SketchGuard (2-phase):     tx/round = {sg/1e6:.2f} MB   per-round {[round(x/1e6,2) for x in stx]}")
    print(f"   SketchGuard breakdown:  sketch {[round(x/1e6,3) for x in ssk]} MB  +  full-fetch {[round(x/1e6,2) for x in sfu]} MB")
    print(f"\n   MEASURED SAVING (SketchGuard vs BALANCE): {100*(1-sg/bal):.1f}%")
