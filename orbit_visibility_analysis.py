#!/usr/bin/env python3
"""
TLE Orbit Visibility Analysis

This script propagates one or more spacecraft using TLE data and the SGP4 model,
computes ground-station visibility passes, generates elevation and azimuth plots,
and produces ground-track visualizations over a user-defined time window.
"""

# orbit_visibility_analysis.py
# Dependencies: pip install skyfield sgp4 matplotlib numpy pandas
# Purpose: Propagate TLE-defined spacecraft orbits, detect ground-station passes,
# generate elevation/azimuth plots, and produce ground-track visualizations.

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
        description="Propagate TLE-defined spacecraft orbits, compute ground-station passes, and generate visibility and ground-track plots."   ) 
    # Input options
    p.add_argument("--tle-file", type=str, help="Path to TLE file with many satellites.")
    p.add_argument("--tle1", type=str, default=DEFAULT_TLE[0], help="TLE line 1 (if not using --tle-file)")
    p.add_argument("--tle2", type=str, default=DEFAULT_TLE[1], help="TLE line 2 (if not using --tle-file)")
    p.add_argument("--name", type=str, default="SAT", help="Name label (single-TLE mode)")
    # Time window
    p.add_argument("--start", type=str, required=True, help="Start ISO8601, e.g. 2025-06-09T00:00:00Z")
    p.add_argument("--end", type=str, required=True, help="End   ISO8601, e.g. 2025-06-10T00:00:00Z")
    # Station
    p.add_argument("--lat", type=float, default=48.123)
    p.add_argument("--lon", type=float, default=9.832)
    p.add_argument("--alt-m", "--alt_m", dest="alt_m", type=float, default=250.0)
    # Pass filter
    p.add_argument("--min-el", "--min_el", dest="min_el", type=float, default=0.0, help="Minimum elevation (deg)")
    return p.parse_args()

# ------------------ Utils ------------------
def iso_to_ts(ts, iso_str):
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    return ts.from_datetime(dt)

def safe_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
    return s if s else "SAT"

def read_tle_file(path: str):
    """
    Parses common TLE file formats:
      name?
      L1
      L2
    (blank lines allowed)
    If no explicit name, use the catalog number from L1 or 'SAT_<idx>'.
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    sats = []
    i = 0
    idx = 1
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            # No name line
            l1, l2 = lines[i], lines[i + 1]
            name = f"SAT_{idx}"
            # try to extract NORAD from line1 cols 3-7 (typical) if present
            m = re.match(r"1\s+(\d+)", l1)
            if m:
                name = f"SAT_{m.group(1)}"
            sats.append((name, l1, l2))
            i += 2
            idx += 1
        elif (i + 2 < len(lines)) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            # Name + two lines
            name = safe_name(lines[i])
            l1, l2 = lines[i + 1], lines[i + 2]
            sats.append((name, l1, l2))
            i += 3
            idx += 1
        else:
            # Skip unrecognized line and move on
            i += 1
    return sats
# ------------------ Ground-track helper ------------------
def plot_ground_track(sat, ts, t0, t1, out_png="groundtrack.png",
                      gs_lat=48.123, gs_lon=9.832, gs_alt_m=250.0, name="SAT"):
    total_s = (t1.utc_datetime() - t0.utc_datetime()).total_seconds()
    n = max(500, int(total_s // 20))  # ~20 s cadence, min 500 points
    t_grid = ts.linspace(t0, t1, n)

    geoc = sat.at(t_grid)
    sp = geoc.subpoint()
    lats = sp.latitude.degrees
    lons = sp.longitude.degrees

    # Split lines at wrap-around to avoid connecting across ±180°
    segments = []
    seg_start = 0
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > 180:
            segments.append((lons[seg_start:i], lats[seg_start:i]))
            seg_start = i
    segments.append((lons[seg_start:], lats[seg_start:]))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{name} Ground Track ({t0.utc_iso()[:16]} → {t1.utc_iso()[:16]} UTC)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_xlim([-180, 180]); ax.set_ylim([-90, 90])
    ax.grid(True, alpha=0.3)

    for mer in range(-180, 181, 30): ax.axvline(mer, color="k", lw=0.2, alpha=0.3)
    for par in range(-90, 91, 30):  ax.axhline(par, color="k", lw=0.2, alpha=0.3)

    for (seg_lon, seg_lat) in segments:
        ax.plot(seg_lon, seg_lat, lw=1.5)

    ax.plot(lons[0],  lats[0],  "o", ms=6, label="Start", zorder=5)
    ax.plot(lons[-1], lats[-1], "s", ms=6, label="End",   zorder=5)
    ax.plot([gs_lon], [gs_lat], "^", ms=7, label="Ground station", zorder=6)

    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# ------------------ core per-satellite workflow ------------------
def process_one_sat(name, tle1, tle2, ts, t0, t1, args):
    """Compute passes, plots, and ground track for a single satellite."""
    sat = EarthSatellite(tle1, tle2, name, ts)
    gs = wgs84.latlon(args.lat, args.lon, elevation_m=args.alt_m)

    # Find rise/transit/set events at min elevation
    t_events, events = sat.find_events(gs, t0, t1, altitude_degrees=args.min_el)

    # Group into passes: rise (0) ... set (2)
    passes = []
    i = 0
    while i < len(events):
        if events[i] == 0:
            j = i + 1
            while j < len(events) and events[j] != 2:
                j += 1
            if j < len(events) and events[j] == 2:
                t_rise = t_events[i]
                t_set = t_events[j]
                t_culm = None
                for k in range(i + 1, j):
                    if events[k] == 1:
                        t_culm = t_events[k]
                        break
                passes.append((t_rise, t_culm, t_set))
                i = j + 1
                continue
        i += 1

    all_elev_series, all_az_series, pass_rows = [], [], []

    for idx, (t_rise, t_culm, t_set) in enumerate(passes, start=1):
        start_dt = t_rise.utc_datetime()
        end_dt = t_set.utc_datetime()
        duration_s = (end_dt - start_dt).total_seconds()

        # sampling per pass
        n = int(min(max(duration_s / 5.0, 60), 2000))
        t_grid = ts.linspace(t_rise, t_set, n)

        # Compute topocentric satellite-to-station geometry at each time sample

        difference = sat.at(t_grid) - gs.at(t_grid)
        alt, az, _ = difference.altaz()

        elev_deg = alt.degrees
        az_deg = az.degrees
        times_iso = np.array([dt.isoformat().replace("+00:00", "Z") for dt in t_grid.utc_datetime()])

        all_elev_series.append((idx, times_iso, elev_deg))
        all_az_series.append((idx, times_iso, az_deg))

        culm_iso = t_culm.utc_datetime().isoformat().replace("+00:00", "Z") if t_culm is not None else ""
        pass_rows.append({
            "sat": name,
            "pass": idx,
            "start_utc": start_dt.isoformat().replace("+00:00", "Z"),
            "end_utc": end_dt.isoformat().replace("+00:00", "Z"),
            "duration_s": round(duration_s, 1),
            "duration_min": round(duration_s / 60.0, 2),
            "culmination_utc": culm_iso
        })

    # save CSV
    df = pd.DataFrame(pass_rows)
    csv_path = f"passes_{safe_name(name)}.csv"
    df.to_csv(csv_path, index=False)
    if not df.empty:
        # elevation plot
        fig1 = plt.figure(figsize=(10, 5))
        for (idx, times, elev_deg) in all_elev_series:
            x = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times]
            dur = df.loc[df['pass'] == idx, 'duration_min'].values[0]
            plt.plot(x, elev_deg, label=f"Pass {idx} ({dur} min)")
        plt.title(f"{name} Elevation vs Time (lat {args.lat:.3f}, lon {args.lon:.3f}, alt {args.alt_m:.0f} m; min_el {args.min_el}°)")
        plt.xlabel("UTC Time"); plt.ylabel("Elevation (deg)")
        plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(f"elevation_{safe_name(name)}.png", dpi=200)
        plt.close(fig1)

        # azimuth plot
        fig2 = plt.figure(figsize=(10, 5))
        for (idx, times, az_deg) in all_az_series:
            x = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times]
            plt.plot(x, az_deg, label=f"Pass {idx}")
        plt.title(f"{name} Azimuth vs Time")
        plt.xlabel("UTC Time"); plt.ylabel("Azimuth (deg, 0=N)")
        plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(f"azimuth_{safe_name(name)}.png", dpi=200)
        plt.close(fig2)

    # ground track always
    plot_ground_track(
        sat=sat, ts=ts, t0=t0, t1=t1,
        out_png=f"groundtrack_{safe_name(name)}.png",
        gs_lat=args.lat, gs_lon=args.lon, gs_alt_m=args.alt_m, name=name
    )

    # Also print a quick summary to terminal
    if df.empty:
        print(f"[{name}] No passes found in the given window.")
    else:
        print(f"[{name}] Passes:")
        print(df.to_string(index=False))

# ------------------ main ------------------
def main():
    args = parse_args()
    ts = load.timescale()
    t0 = iso_to_ts(ts, args.start)
    t1 = iso_to_ts(ts, args.end)

    if args.tle_file:
        sats = read_tle_file(args.tle_file)
        if not sats:
            raise SystemExit(f"No valid TLEs found in: {args.tle_file}")
        for (nm, l1, l2) in sats:
            process_one_sat(nm, l1, l2, ts, t0, t1, args)
    else:
        process_one_sat(args.name, args.tle1, args.tle2, ts, t0, t1, args)

if __name__ == "__main__":
    main()
