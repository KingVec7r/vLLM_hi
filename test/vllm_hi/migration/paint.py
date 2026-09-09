# Reads V0.1\server_test_info_1.json, plots speed_token_per_s versus second from metrics.instantaneous_output_token_speed_token_per_s, and annotates migration_events.event at the correct migration_events.second moments on the chart

# Uses the conda environment "data" to complete the task

import json
import os

import matplotlib

# Ensure an interactive GUI backend (e.g. TkAgg) so plt.show() opens and keeps an interactive window
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt

file_name = "ipc_1_result.json"
out_file_name = "ipc_1.png"
def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data["metrics"]["instantaneous_output_token_speed_token_per_s"]
    seconds = [p["second"] for p in points]
    speeds = [p["speed_token_per_s"] for p in points]
    return seconds, speeds, data


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, file_name)

    seconds, speeds, data = load_data(data_path)
    migration_events = data["migration_events"]
    config = data.get("config", {})

    fig, ax = plt.subplots(figsize=(14, 7))

    # Main curve: instantaneous output token speed over time
    ax.plot(seconds, speeds, linewidth=1.2, color="#1f77b4", label="speed_token_per_s")

    # Leave extra annotation space at the top to avoid the curve being covered by event labels
    y_max = max(speeds)
    label_band_top = y_max * 1.42
    label_band_bottom = y_max * 1.02
    ax.set_ylim(0, label_band_top)
    ax.set_xlim(0, max(seconds) * 1.02)

    def place_event_labels(ax, fig, events, band_bottom, band_top):
        """Assigns horizontal event labels to non-overlapping horizontal lanes to avoid overlaps between events at nearby times.

        Returns [(event, x, y, w, h), ...], where x/y are the data coordinates of the label center and w/h are the text width/height.
        """
        fig.canvas.draw()  # complete layout first to ensure accurate text size measurement
        renderer = fig.canvas.get_renderer()

        # Temporarily place the text to measure its actual rendered width/height
        tmp = []
        for e in events:
            t = ax.text(0, 0, e["event"], rotation=0, fontsize=8, ha="center", va="center")
            tmp.append(t)
        extents = [t.get_window_extent(renderer) for t in tmp]
        for t in tmp:
            t.remove()

        inv = ax.transData.inverted()

        def pix_to_data(pw, ph):
            (x0, y0) = inv.transform((0, 0))
            (x1, _) = inv.transform((pw, 0))
            (_, y1) = inv.transform((0, ph))
            return abs(x1 - x0), abs(y1 - y0)

        widths = []
        heights = []
        for ex in extents:
            w, h = pix_to_data(ex.width, ex.height)
            widths.append(w)
            heights.append(h)

        lane_step = max(heights) * 1.6 if heights else 1.0
        gap = 2.0  # minimum horizontal spacing between labels within the same lane (seconds)
        lanes = []  # track the rightmost x already placed in each lane
        placed = []
        for e, w, h in zip(events, widths, heights):
            x = e["second"]
            left = x - w / 2
            right = x + w / 2
            lane_idx = None
            for i, rightmost in enumerate(lanes):
                if left > rightmost + gap:
                    lane_idx = i
                    break
            if lane_idx is None:
                lane_idx = len(lanes)
                lanes.append(float("-inf"))
            lanes[lane_idx] = max(lanes[lane_idx], right)
            placed.append((e, x, band_top - lane_idx * lane_step, w, h))
        return placed

    events_sorted = sorted(migration_events, key=lambda e: e["second"])
    placed = place_event_labels(ax, fig, events_sorted, label_band_bottom, label_band_top)

    # Draw vertical timestamp dashed lines + horizontal event labels + short leaders
    for e, x, y, w, h in placed:
        # The vertical dashed line only extends to the top of the curve region to avoid crossing label text
        ax.axvline(x, color="#999999", linestyle="--", linewidth=0.9, alpha=0.6,
                   ymin=0, ymax=label_band_bottom / label_band_top)
        # Draw a short line from the bottom of the label center down to the dashed line at the corresponding time
        ax.plot([x, x], [y - h / 2, label_band_bottom],
                color="#999999", linewidth=0.7, alpha=0.6, zorder=1)
        is_key = bool(e.get("is_key_event"))
        ax.text(
            x, y, e["event"],
            rotation=0, fontsize=8, ha="center", va="center",
            color="#c0392b" if is_key else "#34495e",
            fontweight="bold" if is_key else "normal",
            zorder=3,
        )

    # Legend: distinguish key events (red bold) from ordinary events
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=1.2, label="speed_token_per_s"),
        Line2D([0], [0], color="#c0392b", linewidth=0, marker="|", label="key migration event"),
        Line2D([0], [0], color="#34495e", linewidth=0, marker="|", label="migration event"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instantaneous output token speed (token/s)")
    ax.set_title(
        f"Instantaneous Output Token Speed over Time\n"
        f"model={config.get('model')}  concurrency={config.get('concurrency')}  "
        f"migration_delay_s={config.get('migration_delay_s')}  max_tokens={config.get('max_tokens')}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(here, out_file_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

    # Blocking display; keep the interactive window open until the user closes it
    plt.show(block=True)


if __name__ == "__main__":
    main()
