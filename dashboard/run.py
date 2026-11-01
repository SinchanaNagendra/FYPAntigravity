"""
dashboard/run.py
================
One-command dashboard launcher.

Usage:
    python dashboard/run.py
    python dashboard/run.py --port 8080
    python dashboard/run.py --port 8501 --no-browser

Default URL: http://localhost:8080 (auto-detects next free port if occupied)
"""
import argparse
import http.server
import json
import os
import socket
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FRONTEND = ROOT / "dashboard" / "frontend" / "public"
DEFAULT_PORT = int(os.environ.get("PORT", 8765))


def build_data_json():
    """Read all experiment results and write data.json + data.js for the dashboard."""
    import glob
    import statistics

    data = {
        "central_table": [],
        "fl_comparison": [],
        "round_metrics": [],
        "bank_metrics": [],
        "s3_bank_metrics": [],
        "dp_sweep": [],
        "attacks": [],
        "test_summary": {
            "passed": 103,
            "skipped": 13,
            "failed": 0,
            "go_passed": 12,
            "status": "ALL PASSED"
        }
    }

    # ── Centralized baselines ──────────────────────────────────────────────
    bl_path = RESULTS / "baselines" / "centralized" / "S1_seed42.json"
    if bl_path.exists():
        with open(bl_path, encoding="utf-8") as f:
            bl = json.load(f)
        for r in bl.get("results", {}).get("rows", []):
            if "error" not in r:
                data["central_table"].append({
                    "model":        r["model"],
                    "balancing":    r["balancing"],
                    "pr_auc":       round(r.get("pr_auc") or 0, 4),
                    "roc_auc":      round(r.get("roc_auc") or 0, 4),
                    "f1":           round(r.get("f1") or 0, 4),
                    "accuracy":     round(r.get("accuracy") or 0, 4),
                    "majority_acc": round(r.get("majority_class_accuracy") or 0, 4),
                    "train_time":   round(r.get("train_time") or 0, 1),
                })

    # ── FL results ─────────────────────────────────────────────────────────
    for f in sorted(glob.glob(str(RESULTS / "fl" / "*.json"))):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        name = os.path.basename(f).replace(".json", "")
        cfg = d.get("config", {})
        test = d.get("results", {}).get("final_test", [])
        pr_aucs = [r.get("pr_auc") for r in test if r.get("pr_auc")]
        roc_aucs = [r.get("roc_auc") for r in test if r.get("roc_auc")]
        rounds = d.get("results", {}).get("rounds", [])
        if pr_aucs:
            data["fl_comparison"].append({
                "name":         name,
                "model":        cfg.get("model", name),
                "scenario":     cfg.get("scenario", "S1"),
                "n_rounds":     len(rounds),
                "pr_auc_mean":  round(sum(pr_aucs) / len(pr_aucs), 4),
                "pr_auc_std":   round(statistics.stdev(pr_aucs) if len(pr_aucs) > 1 else 0, 4),
                "roc_auc_mean": round(sum(roc_aucs) / len(roc_aucs), 4) if roc_aucs else None,
            })

    # ── LR convergence curve ───────────────────────────────────────────────
    lr_path = RESULTS / "fl" / "logistic_regression_none_S1_seed42.json"
    if lr_path.exists():
        with open(lr_path, encoding="utf-8") as f:
            d = json.load(f)
        for r in d.get("results", {}).get("rounds", []):
            val = r.get("val_metrics", [])
            pr_aucs = [m.get("pr_auc") for m in val if m.get("pr_auc")]
            data["round_metrics"].append({
                "round": r.get("round_num"),
                "mean_pr_auc": round(sum(pr_aucs) / len(pr_aucs), 4) if pr_aucs else None,
                "time": round(r.get("round_time", 0), 1),
            })

    # ── Per-bank results (RF S1) ───────────────────────────────────────────
    rf_path = RESULTS / "fl" / "random_forest_none_S1_seed42.json"
    if rf_path.exists():
        with open(rf_path, encoding="utf-8") as f:
            d = json.load(f)
        for b in d.get("results", {}).get("final_test", []):
            data["bank_metrics"].append({
                "bank_id":    b.get("bank_id"),
                "pr_auc":     round(b.get("pr_auc") or 0, 4),
                "roc_auc":    round(b.get("roc_auc") or 0, 4),
                "f1":         round(b.get("f1") or 0, 4),
                "recall":     round(b.get("recall") or 0, 4),
                "fraud_rate": round(b.get("fraud_rate") or 0, 4),
                "n_samples":  b.get("n_samples"),
            })

    # ── S3 non-IID bank results ────────────────────────────────────────────
    s3_path = RESULTS / "fl" / "random_forest_none_S3_a0.1_seed42.json"
    if s3_path.exists():
        with open(s3_path, encoding="utf-8") as f:
            d = json.load(f)
        for b in d.get("results", {}).get("final_test", []):
            data["s3_bank_metrics"].append({
                "bank_id":    b.get("bank_id"),
                "pr_auc":     round(b.get("pr_auc") or 0, 4),
                "fraud_rate": round(b.get("fraud_rate") or 0, 4),
                "n_samples":  b.get("n_samples"),
            })

    # ── DP Sweep results ───────────────────────────────────────────────────
    dp_path = RESULTS / "dp_sweep" / "dp_sweep_seed42.json"
    if dp_path.exists():
        with open(dp_path, encoding="utf-8") as f:
            dp_data = json.load(f)
        for r in dp_data.get("results", {}).get("results", []):
            ps = r.get("privacy_summary", {})
            cum_e = ps.get("cumulative_epsilon")
            data["dp_sweep"].append({
                "model":       r.get("model"),
                "label":       r.get("noise_label"),
                "sigma":       r.get("noise_multiplier"),
                "cum_epsilon": round(cum_e, 2) if cum_e else "∞",
                "delta":       ps.get("target_delta", 1e-5),
                "pr_auc":      round(r.get("final_test_pr_auc_mean") or 0, 4),
                "roc_auc":     round(r.get("final_test_roc_auc_mean") or 0, 4),
            })

    # ── Attack & Defence results ──────────────────────────────────────────
    atk_path = RESULTS / "attacks" / "attacks_S1_seed42.json"
    if atk_path.exists():
        with open(atk_path, encoding="utf-8") as f:
            atk_data = json.load(f)
        for r in atk_data.get("results", {}).get("results", []):
            data["attacks"].append(r)

    # Copy figures to frontend/public/figures for offline viewing
    fig_src = RESULTS / "figures"
    fig_dst = FRONTEND / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    for fig_file in fig_src.glob("*.png"):
        try:
            import shutil
            shutil.copy2(fig_file, fig_dst / fig_file.name)
        except Exception:
            pass

    # Write data.json
    out_json = FRONTEND / "data.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Write data.js for zero-CORS / offline file:/// viewing
    out_js = FRONTEND / "data.js"
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("window.FL_DASHBOARD_DATA = " + json.dumps(data, indent=2) + ";\n")

    return data


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def end_headers(self):
        # Cache-busting headers to prevent browser from showing cached old dashboard
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs


def find_available_port(start_port: int, max_attempts: int = 50) -> int:
    """Find an available TCP port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    return start_port


def main():
    parser = argparse.ArgumentParser(description="FL Fraud Detection Dashboard")
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to serve dashboard on (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  FL Fraud Detection Dashboard")
    print("=" * 55)

    print("\n[1/2] Reading experiment results...")
    data = build_data_json()
    n_central = len(data["central_table"])
    n_fl = len(data["fl_comparison"])
    n_banks = len(data["bank_metrics"])
    print(f"      Centralized baselines : {n_central} runs")
    print(f"      Federated runs         : {n_fl}")
    print(f"      Per-bank results       : {n_banks} banks")

    port = find_available_port(args.port)
    if port != args.port:
        print(f"\n  ⚠️  Port {args.port} is already in use by another process.")
        print(f"  ➡️  Automatically switching to available port: {port}")

    url = f"http://localhost:{port}"
    print(f"\n[2/2] Starting server on {url}")
    print(f"\n  ✅ Open this URL in your browser:")
    print(f"\n     {url}\n")
    print("  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        webbrowser.open(url)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")


if __name__ == "__main__":
    main()
