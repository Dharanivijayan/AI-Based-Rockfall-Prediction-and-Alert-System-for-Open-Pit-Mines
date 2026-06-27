"""
Synthetic sensor data generator for the AI Rockfall Prediction system.

Generates time-series readings for multiple mine "zones", simulating:
- rainfall (mm/hr)
- vibration / seismic activity (mm/s, PGV-like)
- displacement (mm, cumulative slope movement)
- pore pressure (kPa)

Occasionally injects a "risk spike" pattern (rising rainfall + vibration +
displacement together) to simulate a developing rockfall precursor, so the
ML model and dashboard have something realistic to detect.
"""

import csv
import random
import math
from datetime import datetime, timedelta

ZONES = ["Zone-A", "Zone-B", "Zone-C"]
START = datetime(2026, 1, 1)
INTERVAL_MINUTES = 5
TOTAL_POINTS = 2000  # ~7 days of 5-min data per zone

OUTPUT_FILE = "sensor_data.csv"


def baseline(t_index, seed_offset):
    """Smooth baseline signal with daily cyclic noise."""
    cycle = math.sin((t_index + seed_offset) / 50.0) * 2
    noise = random.uniform(-0.5, 0.5)
    return max(0, cycle + noise)


def generate_zone_series(zone_name, seed_offset):
    rows = []
    displacement = 0.0
    spike_active = False
    spike_countdown = 0

    for i in range(TOTAL_POINTS):
        timestamp = START + timedelta(minutes=INTERVAL_MINUTES * i)

        if not spike_active and random.random() < 0.004:
            spike_active = True
            spike_countdown = random.randint(30, 60)

        rainfall = baseline(i, seed_offset) + (random.uniform(2, 6) if spike_active else 0)
        vibration = baseline(i, seed_offset + 10) + (random.uniform(1, 4) if spike_active else 0)
        pore_pressure = 50 + baseline(i, seed_offset + 20) * 3 + (random.uniform(5, 15) if spike_active else 0)

        increment = random.uniform(0.0, 0.05) + (random.uniform(0.2, 0.6) if spike_active else 0)
        displacement += increment

        rows.append({
            "timestamp": timestamp.isoformat(),
            "zone": zone_name,
            "rainfall_mm_hr": round(rainfall, 2),
            "vibration_mm_s": round(vibration, 2),
            "displacement_mm": round(displacement, 3),
            "pore_pressure_kpa": round(pore_pressure, 2),
        })

        if spike_active:
            spike_countdown -= 1
            if spike_countdown <= 0:
                spike_active = False
                displacement = max(0, displacement - random.uniform(0, 3))

    return rows


def main():
    all_rows = []
    for idx, zone in enumerate(ZONES):
        all_rows.extend(generate_zone_series(zone, seed_offset=idx * 17))

    all_rows.sort(key=lambda r: r["timestamp"])

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Generated {len(all_rows)} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
