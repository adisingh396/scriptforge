"""Live training status for ScriptForge.

    python status.py                  # one snapshot
    python status.py --watch          # refresh every 10s (Ctrl-C to stop)
    python status.py --log logs/train_iam.csv --samples outputs/samples_iam

Shows: latest step, throughput, ETA, loss trends, health warnings, newest sample grid.
"""

import argparse
import os
import re
import time
from datetime import datetime, timedelta


def parse_log(path):
    if not os.path.exists(path):
        return [], []
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            try:
                rows.append({k: (int(float(v)) if k == "step" else float(v))
                             for k, v in zip(header, parts)})
            except ValueError:
                continue
    return header, rows


def parse_train_stdout(path, n=12):
    """Pick the most recent 'step X/Y ...' lines from a captured log, if present."""
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-400:]
    except Exception:
        return []
    hits = [ln.strip() for ln in lines if re.search(r"\bstep \d+/\d+\b", ln)]
    return hits[-n:]


def newest_sample(dirpath, pattern=r"step_(\d+)\.png"):
    if not dirpath or not os.path.isdir(dirpath):
        return None, None
    best, best_step = None, -1
    for name in os.listdir(dirpath):
        m = re.match(pattern, name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best, best_step = os.path.join(dirpath, name), step
    return best, best_step


def fmt_eta(seconds):
    if seconds <= 0 or seconds > 86400 * 7:
        return "?"
    return str(timedelta(seconds=int(seconds)))


def snapshot(cfg):
    print("=" * 78)
    print(f"  ScriptForge status   {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 78)

    header, rows = parse_log(cfg.log)
    if not rows:
        print(f"  no rows in {cfg.log} yet (training may still be starting up)")
    else:
        first, last = rows[0], rows[-1]
        step, first_step = last["step"], first["step"]
        mins = last.get("elapsed_min", 0.0) - first.get("elapsed_min", 0.0)
        spm = (step - first_step) / mins if mins > 0 else 0.0
        print(f"  log          : {cfg.log}  ({len(rows)} rows)")
        print(f"  step         : {int(step)}/{cfg.steps}   " +
              (f"({100.0 * step / cfg.steps:.1f}%)" if cfg.steps else ""))
        print(f"  speed        : {spm:.0f} steps/min ({spm / 60.0:.2f}/s)   " +
              (f"ETA {fmt_eta((cfg.steps - step) / spm * 60)}" if spm > 0 and cfg.steps else ""))
        print(f"  elapsed      : {last.get('elapsed_min', 0):.1f} min")

        print("-" * 78)
        keys = [k for k in ("d_loss", "g_loss", "adv", "l1", "ctc", "d_real", "d_fake") if k in last]
        print("  current      : " + "  ".join(f"{k}={last[k]:.3f}" for k in keys))
        if len(rows) > 20:
            ref = rows[max(0, len(rows) - 60)]
            print("  ~60 rows ago : " + "  ".join(f"{k}={ref[k]:.3f}" for k in keys))

        # health checks that catch the failures hit during development
        if "d_real" in last and "d_fake" in last:
            gap = last["d_real"] - last["d_fake"]
            if abs(gap) < 0.02:
                print("  !! WARNING: D cannot separate real from fake (d_real ~= d_fake).")
                print("     The adversarial signal is dead -> output will look like a traced font.")
                print("     Fix: train with --adv bce --sn 0 and --r1-weight 0.")
            elif gap < 0:
                print("  !! note: d_fake > d_real (D inverted) — usually transient early on.")
            elif last["d_real"] > 6:
                print("  !! note: D is very strong; generator may lose the adversarial signal.")
        if "ctc" in last and cfg.ctc_weight and cfg.ctc_weight > 0:
            if last["ctc"] > 1.5:
                print(f"  !! WARNING: CTC loss {last['ctc']:.2f} — words are not readable yet.")
            else:
                print(f"  ok: generated words are recognised (CTC {last['ctc']:.2f}); lower is better.")
        if "l1" in last and last["l1"] > 0.30:
            print(f"  note: L1 {last['l1']:.2f} still high — shape not learned yet.")

    print("-" * 78)
    latest, step_seen = newest_sample(cfg.samples)
    if latest:
        age = (time.time() - os.path.getmtime(latest)) / 60.0
        print(f"  newest grid  : {latest}  (step {step_seen}, {age:.1f} min old)")
        print("                 open this PNG to eyeball quality, e.g.:")
        print(f'                 start "" "{latest}"')
    else:
        print(f"  newest grid  : none in {cfg.samples}")
    ckpts = sorted([p for p in (cfg.ckpt or "",) if os.path.exists(p)]) if cfg.ckpt else []
    if cfg.ckpt and os.path.exists(cfg.ckpt):
        print(f"  checkpoint   : {cfg.ckpt}  "
              f"({os.path.getsize(cfg.ckpt) / 1e6:.1f} MB, "
              f"{(time.time() - os.path.getmtime(cfg.ckpt)) / 60:.1f} min old)")
    if cfg.stdout and os.path.exists(cfg.stdout):
        tail = parse_train_stdout(cfg.stdout, 4)
        if tail:
            print("  last stdout  :")
            for ln in tail:
                print("     " + ln[:110])
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/train_iam.csv")
    ap.add_argument("--samples", default="outputs/samples_iam")
    ap.add_argument("--ckpt", default="checkpoints/gan_iam.pt")
    ap.add_argument("--steps", type=int, default=0, help="total steps you launched with (for ETA)")
    ap.add_argument("--ctc-weight", type=float, default=0.5,
                    help="set 0 if you launched the GAN without the CTC legibility loss")
    ap.add_argument("--stdout", default="", help="optional captured training stdout file")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()
    try:
        while True:
            snapshot(args)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped watching.")


if __name__ == "__main__":
    main()
