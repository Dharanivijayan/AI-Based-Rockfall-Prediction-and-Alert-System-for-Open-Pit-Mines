# AI-Based Rockfall Prediction and Alert System for Open-Pit Mines

**SIH 2026 — Problem Statement ID: 25071**
Category: Software | Theme: Disaster Management
Organization: Ministry of Mines | Department: National Institute of Rock Mechanics (NIRM)

## Overview

This is the software prototype for an AI-driven early-warning system that
predicts rockfall risk in open-pit mines using multi-source sensor data
(rainfall, vibration, displacement, pore pressure) and issues tiered alerts
(**ADVISORY → WARNING → EVACUATE**) before a slope failure occurs.

Since the Problem Statement category is **Software**, this repo assumes
sensor hardware (radar, geotechnical sensors, etc.) already exists at the
mine site. We simulate that hardware layer with a synthetic data generator
so the prediction engine, API, and dashboard can be demonstrated end-to-end
without physical equipment.

## Architecture
