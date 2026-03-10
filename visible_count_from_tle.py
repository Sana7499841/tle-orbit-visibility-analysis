#!/usr/bin/env python3
"""
Visible Satellite Count from TLE Data

This script propagates one or more spacecraft using TLE data and computes the
number of satellites simultaneously visible from a ground station above a
specified elevation mask over a user-defined time window.

Outputs:
- visible_count.csv
- visible_count.png
"""
# visible_count_from_tle.py
# Dependencies: pip install skyfield sgp4 matplotlib numpy pandas

import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84, EarthSatellite

DEFAULT_TLE = [
    "1 64056U 25104B   25160.24306210  .00859907  25185-3 17582-2 0  9992",
    "2 64056  41.9357 156.0687 0193223  48.4945 313.2311 15.73238515 3578",
]

# ------------------ CLI ------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute the number of satellites simultaneously visible from a ground station over time."
    )
    # Input options
    p.add_argument("--tle-file", type=str, help="Path to a TLE file (multiple satellites).")
    p.add_argument("--tle1", type=str, default=DEFAULT_TLE[0], help="TLE line 1 (single-sat mode)")
    p.add_argument("--tle2", type=str, default=DEFAULT_TLE[1], help="TLE line 2 (single-sat mode)")
    p.add_argument("--name", type=str, default="SAT", help="Name label (single-sat mode)")

    # Time window
    p.add_argument("--start", type=str, required=True, help="Start ISO8601, e.g. 2025-06-09T00:00:00Z")
    p.add_argument("--end",   type=str, required=True, help="End   ISO8601, e.g. 2025-06-11T00:00:00Z")

    # Ground station
    p.add_argument("--lat", type=float, default=48.123)
    p.add_argument("--lon", type=float, default=9.832)
    p.add_argument("--alt-m", "--alt_m", dest="alt_m", type=float, default=250.0)

    # Visibility criteria and sampling
    p.add_argument("--min-el", "--min_el", dest="min_el", type=float, default=0.0,
                   help="Minimum elevation (deg) to count as 'visible'")
    p.add_argument("--step-sec", type=int, default=60,
                   help="Sampling step (seconds) for counting (default 60s)")

    # Optional: only take the first N satellites from the file (speeds things up)
    p.add_argument("--subset", type=int, default=0,
                   help="If >0, only use the first N satellites from --tle-file")
    return p.parse_args()

# ------------------ Utils ------------------
def iso_to_ts(ts, iso_str):
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    return ts.from_datetime(dt)

def safe_name(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
    return s or "SAT"

def read_tle_file(path: str):
    """
    Accepts files like:
      Name (optional)
      L1
      L2
    or:
      L1
      L2
    (blank lines allowed)
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    sats = []
    i, idx = 0, 1
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            l1, l2 = lines[i], lines[i + 1]
            name = f"SAT_{idx}"
            m = re.match(r"1\s+(\d+)", l1)
            if m: name = f"SAT_{m.group(1)}"
            sats.append((name, l1, l2))
            i += 2; idx += 1
        elif (i + 2 < len(lines)) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name = safe_name(lines[i]); l1, l2 = lines[i + 1], lines[i + 2]
            sats.append((name, l1, l2))
            i += 3; idx += 1
        else:
            i += 1
    return sats

# ------------------ Visible count ------------------
def compute_visible_count_over_time(sat_list, gs, ts, t0, t1, min_el_deg, step_sec=60):
    """Return list[datetime], np.array[int] with count of sats whose elevation >= min_el at each time."""
    total_s = (t1.utc_datetime() - t0.utc_datetime()).total_seconds()
    n = max(2, int(total_s // step_sec) + 1)
    t_grid = ts.linspace(t0, t1, n)

    counts = np.zeros(n, dtype=int)
    for sat in sat_list:
        diff = sat.at(t_grid) - gs.at(t_grid)
        alt, _, _ = diff.altaz()
        counts += (alt.degrees >= min_el_deg).astype(int)

    return list(t_grid.utc_datetime()), counts

# ------------------ main ------------------
def main():
    args = parse_args()
    if args.step_sec <= 0:
         raise SystemExit("--step-sec must be a positive integer")
    ts = load.timescale()
    t0 = iso_to_ts(ts, args.start)
    t1 = iso_to_ts(ts, args.end)

    # Ground station
    gs = wgs84.latlon(args.lat, args.lon, elevation_m=args.alt_m)

    # Build list of satellites to count
    sat_list = []
    if args.tle_file:
        sats = read_tle_file(args.tle_file)
        if args.subset and args.subset > 0:
            sats = sats[:args.subset]
        if not sats:
            raise SystemExit(f"No valid TLEs found in: {args.tle_file}")
        for (nm, l1, l2) in sats:
            sat_list.append(EarthSatellite(l1, l2, nm, ts))
        print(f"Loaded {len(sat_list)} satellites from {args.tle_file}")
    else:
        sat_list.append(EarthSatellite(args.tle1, args.tle2, args.name, ts))
        print(f"Loaded 1 satellite (single-TLE mode)")

    # Count visible satellites vs time
    times_dt, counts = compute_visible_count_over_time(
        sat_list, gs, ts, t0, t1, min_el_deg=args.min_el, step_sec=args.step_sec
    )

    # Save CSV
    vis_df = pd.DataFrame({
        "utc": [dt.isoformat().replace("+00:00", "Z") for dt in times_dt],
        "visible_count": counts
    })
    vis_df.to_csv("visible_count.csv", index=False)

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(times_dt, counts, linewidth=1.8)
    plt.title(f"Number of Visible Satellites vs Time (Elevation ≥ {args.min_el}°, Step {args.step_sec}s)")
    plt.xlabel("UTC Time")
    plt.ylabel("Visible satellites (elev ≥ min_el)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visible_count.png", dpi=200)
    plt.close()
    print("Wrote visible_count.csv and visible_count.png")

if __name__ == "__main__":
    main()
