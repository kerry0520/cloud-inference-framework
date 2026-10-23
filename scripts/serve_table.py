"""Render a serve-stage summary.json as markdown tables for the stage record."""

import argparse
import json
from pathlib import Path


def ms(value):
    return "—" if value is None else "%.1f" % value


def box(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    checks = summary["checks"]
    print("checks: " + json.dumps(checks, sort_keys=True))
    print("verdict: " + summary.get("verdict", "?"))
    if summary.get("token_agreement"):
        print("token agreement: " + json.dumps(summary["token_agreement"]))
    print()

    rows = []
    for arm in ("base", "patched"):
        for phase in ("nop", "minp"):
            data = summary["arms"][arm]["phases"][phase]
            rows.append([
                arm, phase, data["ttft_ms"]["n"],
                ms(data["ttft_ms"]["median"]), ms(data["ttft_ms"]["p99"]),
                ms(data["tpot_ms"]["median"]), ms(data["tpot_ms"]["p99"]),
                ms(data["itl_median_ms"]["median"]),
                ms(data["total_ms"]["median"]),
                "%.0f" % data["requests_per_second"]["median"],
                "%.0f" % data["tokens_per_second"]["median"],
            ])
    print(box(["arm", "phase", "n", "TTFT p50", "TTFT p99", "TPOT p50", "TPOT p99",
               "ITL p50", "总时延 p50", "req/s", "tok/s"], rows))
    print()

    for phase in ("nop", "minp"):
        rows = []
        for entry in summary["paired"]:
            data = entry[phase]
            rows.append([
                entry["block"],
                "%.1f → %.1f" % (data["ttft_ms"]["base"], data["ttft_ms"]["patched"]),
                "%+.1f%%" % data["ttft_ms"]["delta_pct"],
                "%.2f → %.2f" % (data["tpot_ms"]["base"], data["tpot_ms"]["patched"]),
                "%+.2f%%" % data["tpot_ms"]["delta_pct"],
                "%.0f → %.0f" % (data["total_ms"]["base"], data["total_ms"]["patched"]),
                "%+.2f%%" % data["total_ms"]["delta_pct"],
                "%+.2f%%" % data["tokens_per_second"]["delta_pct"],
            ])
        print("phase=%s（每块 base 两场的会话中位数 vs patched 两场）" % phase)
        print(box(["block", "TTFT ms", "ΔTTFT", "TPOT ms", "ΔTPOT", "总时延 ms",
                   "Δ总时延", "Δtok/s"], rows))
        print()


if __name__ == "__main__":
    main()
