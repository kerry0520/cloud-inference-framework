"""Render the three summary figures as dependency-free SVG.

The charts are deliberately simple: one decomposition bar, one byte-time
scatter with the one-parameter model, and one local-versus-request comparison.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width, height, title):
        self.width, self.height, self.title = width, height, title
        self.parts = []

    def rect(self, x, y, w, h, fill, stroke="none", width=0.0, opacity=1.0):
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>')

    def circle(self, x, y, radius, fill):
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}"/>')

    def line(self, x1, y1, x2, y2, stroke="#333", width=1.0, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{stroke}" stroke-width="{width}"{extra}/>')

    def text(self, x, y, body, size=12, anchor="start", fill="#111", weight="normal",
             rotate=None):
        extra = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI, Arial" '
                          f'font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
                          f'font-weight="{weight}"{extra}>{escape(body)}</text>')

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
                f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">'
                f'<rect width="100%" height="100%" fill="#ffffff"/>')
        self.parts.insert(0, head)
        self.parts.append(f'<text x="{self.width / 2:.0f}" y="26" font-family="Segoe UI, Arial" '
                          f'font-size="16" text-anchor="middle" fill="#111" font-weight="bold">'
                          f'{escape(self.title)}</text>')
        Path(path).write_text("\n".join(self.parts) + "\n</svg>\n")


def figure_breakdown(attribution, path):
    roles = attribution["per_role"]
    keep = ("qkv_proj", "o_proj", "gate_up", "down_proj", "lm_head", "attention")
    items = [(name, roles[name]["per_step_us"]) for name in keep]
    other = sum(entry["per_step_us"] for name, entry in roles.items()
                if name not in keep and name != "final_norm")
    items.append(("everything else", other))
    items.sort(key=lambda pair: pair[1])
    total = attribution["median_step_kernel_sum_us"]
    svg = Svg(860, 70 + 28 * len(items), "Where a batch-2 decode step spends its time")
    left, right = 190, 690
    largest = max(value for _, value in items)
    for index, (name, micros) in enumerate(items):
        y = 62 + index * 28
        width = (right - left) * micros / largest
        solid = name in ("qkv_proj", "o_proj", "gate_up", "down_proj", "lm_head")
        svg.rect(left, y - 14, width, 19, "#3f6f9f" if solid else "#a9bfd4")
        svg.text(left - 8, y, name, size=12, anchor="end")
        svg.text(left + width + 6, y, f"{micros:.0f} us  {100 * micros / total:.1f}%", size=11)
    svg.text(left, svg.height - 10,
             f"total kernel time per step {total:.0f} us (instrumented trace, batch 2)",
             size=11, fill="#555")
    svg.save(path)


def figure_roofline(roofline, path):
    rows = roofline["roles"]
    fitted = roofline["fitted_effective_GB_s"]
    svg = Svg(660, 440, "Weight bytes predict time (one fitted constant)")
    left, right, top, bottom = 92, 600, 56, 340
    xmin, xmax = math.log10(3e7), math.log10(4e8)
    ymin, ymax = math.log10(300), math.log10(1200)

    def px(value):
        return left + (math.log10(value) - xmin) / (xmax - xmin) * (right - left)

    def py(value):
        return bottom - (math.log10(value) - ymin) / (ymax - ymin) * (bottom - top)

    for label, value in (("30 MB", 3e7), ("100 MB", 1e8), ("300 MB", 3e8)):
        x = px(value)
        svg.line(x, top, x, bottom, "#e2e2e2")
        svg.text(x, bottom + 16, label, size=10, anchor="middle", fill="#555")
    for value in (300, 400, 600, 800, 1000):
        y = py(value)
        svg.line(left, y, right, y, "#ececec")
        svg.text(left - 6, y + 4, str(value), size=10, anchor="end", fill="#555")
    svg.line(left, bottom, right, bottom, "#333", 1.2)
    svg.line(left, top, left, bottom, "#333", 1.2)
    svg.text((left + right) / 2, bottom + 38, "weight bytes streamed per step (log)",
             size=11, anchor="middle")
    svg.text(24, (top + bottom) / 2, "kernel time, us (log)", size=11, anchor="middle",
             rotate=-90)

    start_x = max(3e7, 300 * fitted * 1e3)
    end_x = min(4e8, 1200 * fitted * 1e3)
    svg.line(px(start_x), py(start_x / (fitted * 1e3)),
             px(end_x), py(end_x / (fitted * 1e3)), "#c0392b", 1.2, "6,4")
    for row in rows:
        x, y = px(row["step_bytes"]), py(row["measured_us"])
        svg.circle(x, y, 4.2, "#1f4e79")
        anchor = "end" if row["step_bytes"] > 2.5e8 else "start"
        offset = -10 if anchor == "end" else 10
        svg.text(x + offset, y - 5, row["role"], size=10, anchor=anchor)
        svg.text(x + offset, y + 8, f"{row['residual_pct']:+.0f}%", size=9,
                 anchor=anchor, fill="#a93a2f")
    svg.text(left + 8, top - 14,
             f"dashed: bytes / {fitted:.0f} GB/s; red: leave-one-out residual",
             size=10, fill="#555", anchor="start")
    svg.save(path)


def figure_local_vs_request(summary, micro_path, path):
    samples = [[], []]
    for line in Path(micro_path).read_text().splitlines():
        row = json.loads(line)
        if row["vocab"] == 151936 and row["batch"] == 2:
            samples[0 if row["arm"] == "base" else 1].append(row["median_us"])
    values = []
    if all(samples):
        values = [("upstream", statistics.median(samples[0])),
                  ("patched", statistics.median(samples[1]))]
    svg = Svg(760, 470, "A 10x kernel win is a ~1% request win")
    left = 110
    svg.text(left, 56, "(a) min-p kernel, V151936 batch 2 (microbenchmark)",
             size=11, weight="bold")
    bar_left, pixels_per_us = 210, 2.4
    for index, (label, micros) in enumerate(values):
        y = 84 + index * 40
        width = micros * pixels_per_us
        svg.rect(bar_left, y - 16, width, 24, "#1f4e79" if index else "#8fa8bf")
        svg.text(bar_left - 8, y, label, size=11, anchor="end")
        svg.text(bar_left + width + 8, y, f"{micros:.1f} us", size=11)

    svg.text(left, 200, "(b) whole request wall time change vs upstream, 95% CI (paired rounds)",
             size=11, weight="bold")
    zero = 420
    svg.line(zero, 224, zero, 420, "#333", 1.2)
    svg.text(zero, 436, "0%", size=10, anchor="middle")
    svg.text(zero - 90, 436, "faster", size=10, fill="#1f7a3f")
    svg.text(zero + 90, 436, "slower", size=10, fill="#a00")
    pixels_per_pct = 62
    for index, entry in enumerate(summary.get("summary", [])):
        y = 246 + index * 40
        stats = entry["wall_pct_of_paired_rounds"]
        low, high = stats["ci95"]
        svg.line(zero + low * pixels_per_pct, y, zero + high * pixels_per_pct, y, "#c0392b", 2.0)
        for edge in (low, high):
            x = zero + edge * pixels_per_pct
            svg.line(x, y - 6, x, y + 6, "#c0392b", 2.0)
        svg.rect(zero + stats["median"] * pixels_per_pct - 3, y - 3, 6, 6, "#c0392b")
        svg.text(left, y + 4, f"batch {entry['batch']}", size=11, anchor="start")
        svg.text(zero + 200, y + 4,
                 f"{stats['median']:+.2f}%  [{low:+.2f}, {high:+.2f}]", size=10)
    svg.save(path)


def figure_scaling(runs, path):
    batches = [1, 2, 4, 8]
    svg = Svg(700, 410, "Request-level saving dilutes as the decode step grows")
    left, right, top, bottom = 92, 640, 60, 300
    ymax, ymin = 0.4, -3.0

    def px(batch):
        index = batches.index(batch)
        return left + 60 + index * (right - left - 60) / (len(batches) - 1)

    def py(value):
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    for value in (0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0):
        y = py(value)
        svg.line(left, y, right, y, "#e6e6e6" if value else "#999")
        svg.text(left - 6, y + 4, f"{value:+.1f}%", size=10, anchor="end", fill="#555")
    for batch in batches:
        svg.text(px(batch), bottom + 18, f"batch {batch}", size=10, anchor="middle", fill="#555")
    svg.text((left + right) / 2, bottom + 40, "decode saving vs upstream, paired rounds (95% CI)",
             size=10, anchor="middle", fill="#555")

    predicted = {}
    for _label, summary in runs:
        for entry in summary.get("summary", []):
            decode_ms = entry.get("base_decode_ms_median") or entry.get("base_decode_ms")
            if decode_ms is None:
                details = entry.get("rounds_detail") or []
                decode_ms = statistics.mean(row["base_decode_ms"] for row in details) if details else None
            if decode_ms:
                predicted[entry["batch"]] = -56.9 / (decode_ms * 1000) * 100
    predicted = sorted(predicted.items())
    if len(predicted) >= 2:
        points = " ".join(f"{px(batch):.1f},{py(value):.1f}" for batch, value in predicted)
        svg.parts.append(f'<polyline points="{points}" fill="none" stroke="#555" '
                         f'stroke-width="1.4" stroke-dasharray="6,4"/>')
        svg.text(px(predicted[-1][0]) - 6, py(predicted[-1][1]) - 10,
                 "-56.9 us / decode step", size=10, anchor="end", fill="#555")

    colours = ["#1f4e79", "#c0392b"]
    for series, (label, summary) in enumerate(runs):
        for entry in summary.get("summary", []):
            stats = entry["decode_pct_of_paired_rounds"]
            low, high = stats["ci95"]
            x = px(entry["batch"]) + (series - 0.5) * 14
            svg.line(x, py(max(low, -3.0)), x, py(min(high, 0.35)), colours[series], 2.0)
            for edge in (low, high):
                clipped = max(min(edge, 0.35), -3.0)
                svg.line(x - 5, py(clipped), x + 5, py(clipped), colours[series], 2.0)
            svg.circle(x, py(stats["median"]), 4.2, colours[series])
        svg.rect(right - 150, top + series * 18 - 10, 12, 12, colours[series])
        svg.text(right - 132, top + series * 18 + 1, label, size=10)
    svg.save(path)


def pick_abba(results_dir, prefix):
    """Newest summary.json whose directory name starts with prefix."""
    for directory in reversed(sorted(Path(results_dir).glob(prefix + "*"))):
        candidate = directory / "summary.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())
    return None


def latest(results_dir, pattern, filename):
    for directory in reversed(sorted(Path(results_dir).glob(pattern))):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no {filename} under {pattern} in {results_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--attribution", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--micro", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    figure_breakdown(json.loads(Path(args.attribution).read_text()), out / "step-breakdown.svg")
    figure_roofline(json.loads(latest(args.results, "roofline-*", "roofline.json").read_text()),
                    out / "bytes-vs-time.svg")
    first = pick_abba(args.results, "abba-2026")
    if first is not None:
        figure_local_vs_request(first, args.micro, out / "local-vs-request.svg")
    second = pick_abba(args.results, "abba-scale-2026")
    if first is not None and second is not None:
        figure_scaling([("run A (10-17)", first), ("run B (10-18)", second)],
                       out / "saving-vs-batch.svg")
    print(json.dumps({"figures": sorted(p.name for p in out.glob("*.svg"))}))


if __name__ == "__main__":
    main()
