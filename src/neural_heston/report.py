"""Builds the validation report: report/report.md, whose {{table:...}} placeholders are filled from report/results.json,
rendered to report/validation_report.html and printed to PDF by a headless Edge or Chrome (no LaTeX needed).

  python -m neural_heston.report"""
import base64
import json
import os
import re
import shutil
import subprocess

import markdown

from .validate import GREEKS, REPORT

PARAMS = ("kappa", "theta", "sigma", "rho", "v0")
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "msedge", "google-chrome", "chromium"]
LABELS = {"v0_plain": "v0, price only", "v0_dml": "v0, differential", "plain": "control variate, price only",
          "dml": "control variate, differential"}


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["--:"] * (len(head) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def e(v):
    return f"{v:.1e}".replace("e-0", "e-")


def pct(v):
    return f"{100 * v:.2f} %"


def median(v):
    v = sorted(v)
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2


def t_ablation(R):
    rows = []
    for name in [n for n in LABELS if n in R]:
        r = R[name]
        a, g, arb = r["accuracy"]["in_domain"]["all"], r["greeks"], r["arbitrage"]["network"]
        rows.append([LABELS[name], e(a["price"]["rmse"]), e(a["price"]["max"]), f'{a["iv_bp"]["mae"]:.1f}',
                     f'{a["iv_bp"]["p99"]:.1f}', pct(arb["butterfly"]), pct(arb["calendar"]),
                     pct(g["delta"]["rel_rmse"]), pct(g["gamma"]["rel_rmse"]), pct(g["vega_v0"]["rel_rmse"])])
    return table(["model", "price RMSE", "price max", "IV MAE (bp)", "IV p99 (bp)", "butterfly viol.",
                  "calendar viol.", "delta err.", "gamma err.", "vega err."], rows)


def t_slices(R, name="dml"):
    rows = []
    for s, v in R[name]["accuracy"]["in_domain"].items():
        rows.append([s, v["price"]["n"], e(v["price"]["rmse"]), e(v["price"]["max"]), f'{v["iv_bp"]["mae"]:.1f}',
                     f'{v["iv_bp"]["p99"]:.1f}', f'{v["iv_bp"]["max"]:.0f}'])
    return table(["slice", "n", "price RMSE", "price max", "IV MAE (bp)", "IV p99 (bp)", "IV max (bp)"], rows)


def t_ood(R):
    names = [n for n in ("dml", "plain") if n in R]
    ref = R[names[0]]["accuracy"]["in_domain"]["all"]["price"]["rmse"]
    rows = [["inside the box"] + [e(R[n]["accuracy"]["in_domain"]["all"]["price"]["rmse"]) for n in names] + ["1.0"]]
    for box, v in R[names[0]]["accuracy"]["out_of_domain"].items():
        rows.append([box] + [e(R[n]["accuracy"]["out_of_domain"][box]["price"]["rmse"]) for n in names]
                    + [f'{v["price"]["rmse"] / ref:.1f}'])
    return table(["region"] + [f"{LABELS[n]} RMSE" for n in names] + ["× in-box (differential)"], rows)


def t_arbitrage(R):
    rows = []
    for name in [n for n in ("dml", "plain") if n in R] + ["reference"]:
        a = R["dml"]["arbitrage"]["reference"] if name == "reference" else R[name]["arbitrage"]["network"]
        rows.append([LABELS.get(name, "reference pricer"), f'{a["points"]:,}', pct(a["call_spread"]),
                     pct(a["butterfly"]), e(a["butterfly_worst"]), pct(a["calendar"]), e(a["calendar_worst"])])
    return table(["pricer", "points", "call spread viol.", "butterfly viol.", "worst butterfly", "calendar viol.",
                  "worst calendar"], rows)


def t_greeks(R):
    names = [n for n in ("dml", "plain") if n in R]
    rows = [[gk, e(R[names[0]]["greeks"][gk]["scale"])] + [pct(R[n]["greeks"][gk]["rel_rmse"]) for n in names]
            for gk in GREEKS]
    return table(["Greek", "typical size (std)"] + [f"{LABELS[n]} RMSE / std" for n in names], rows)


def t_calibration(R):
    rows = R["dml"]["calibration"]
    out = []
    for who in ("network", "reference"):
        fit = [r[who]["fit_bp"] for r in rows]
        rep = [r[who]["reprice_bp"] for r in rows]
        sec = [r[who]["seconds"] for r in rows]
        out.append([who, f"{median(fit):.1f}", f"{max(fit):.1f}", f"{median(rep):.1f}", f"{max(rep):.1f}",
                    f"{median(sec):.2f}"])
    return table(["pricer in the loop", "fit median (bp)", "fit max (bp)", "reprice median (bp)",
                  "reprice max (bp)", "seconds (median)"], out)


def t_calibration_params(R):
    rows = R["dml"]["calibration"]
    out = []
    for i, p in enumerate(PARAMS):
        cells = [p]
        for who in ("network", "reference"):
            err = [abs(r[who]["params"][i] - r["truth"][i]) for r in rows]
            cells += [f"{median(err):.3f}", f"{max(err):.3f}"]
        out.append(cells)
    return table(["parameter", "network median abs. error", "network max abs. error", "reference median abs. error",
                  "reference max abs. error"], out)


def t_speed(R):
    rows = [[f"{int(n):,}", f'{v["network_s"] * 1e3:.2f}', f'{v["reference_s"] * 1e3:.1f}', f'{v["speedup"]:.0f}×']
            for n, v in R["dml"]["speed"].items()]
    return table(["batch size", "network (ms)", "reference (ms)", "speed-up"], rows)


def t_training(R):
    rows = []
    for name in [n for n in LABELS if n in R]:
        m = R[name]["meta"]
        rows.append([LABELS[name], f'{m.get("n", 0):,}', m.get("epochs", ""), f'{m.get("train_seconds", 0) / 60:.0f} min'])
    return table(["model", "training points", "epochs", "CPU time"], rows)


TABLES = {"ablation": t_ablation, "slices": t_slices, "ood": t_ood, "arbitrage": t_arbitrage, "greeks": t_greeks,
          "calibration": t_calibration, "calibration_params": t_calibration_params, "speed": t_speed,
          "training": t_training}

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; margin: 0 0 4pt; } h2 { font-size: 13.5pt; margin-top: 18pt; border-bottom: 1px solid #999; }
h3 { font-size: 11pt; margin-top: 12pt; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 8.5pt; font-family: Arial, sans-serif; }
th, td { border-bottom: 1px solid #ccc; padding: 2.5pt 6pt; } th { background: #f0f0f0; }
img { max-width: 100%; display: block; margin: 6pt auto; } code { font-size: 9pt; }
blockquote { border-left: 3px solid #2563eb; margin: 8pt 0; padding: 2pt 10pt; background: #f5f8ff; }
"""


def inline_images(html):
    def sub(mt):
        path = os.path.join(REPORT, mt.group(1))
        if not os.path.exists(path):
            print("missing figure", mt.group(1), "(run neural_heston.figures)")
            return mt.group(0)
        with open(path, "rb") as f:
            return f'src="data:image/png;base64,{base64.b64encode(f.read()).decode()}"'
    return re.sub(r'src="([^"]+\.png)"', sub, html)


def browser():
    for b in BROWSERS:
        if os.path.exists(b) or shutil.which(b):
            return b
    return None


def main():
    with open(os.path.join(REPORT, "results.json")) as f:
        R = json.load(f)
    with open(os.path.join(REPORT, "report.md"), encoding="utf-8") as f:
        md = f.read()
    md = re.sub(r"\{\{table:(\w+)\}\}", lambda mt: TABLES[mt.group(1)](R), md)
    body = inline_images(markdown.markdown(md, extensions=["tables"]))
    html_path = os.path.join(REPORT, "validation_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f'<!doctype html><html><head><meta charset="utf-8"><title>Neural Heston: validation report</title>'
                f"<style>{CSS}</style></head><body>{body}</body></html>")
    b = browser()
    if b is None:
        print("written", html_path, "(no Edge/Chrome found for the PDF)")
        return
    pdf = os.path.abspath(os.path.join(REPORT, "validation_report.pdf"))
    subprocess.run([b, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "file:///" + os.path.abspath(html_path).replace("\\", "/")], check=True, capture_output=True)
    print("written", pdf)


if __name__ == "__main__":
    main()
