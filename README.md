# TLE Orbit Visibility Analysis

Python implementation of satellite orbit propagation using **Two-Line Element (TLE)** data and the **SGP4 orbit propagator**.  
The project performs satellite pass prediction, elevation analysis, and multi-satellite visibility estimation for a specified ground station.

This repository demonstrates how publicly available orbital element data can be used to analyze satellite visibility and pass geometry from a ground observer.

---

## Project Objectives

The goal of this project is to:

- Propagate satellite orbits using **SGP4**
- Predict **satellite passes over a ground station**
- Compute **satellite elevation versus time**
- Estimate **multi-satellite visibility counts**
- Visualize visibility trends for large LEO satellite constellations

---

## Ground Station Parameters

Location used for analysis:

Latitude: **48.123° N**  
Longitude: **9.832° E**  
Altitude: **250 m**

Elevation mask applied in visibility analysis: **10°**

---

## Repository Structure
tle-orbit-visibility-analysis
│
├── orbit_visibility_analysis.py # Satellite pass prediction and elevation analysis
├── visible_count_from_tle.py # Multi-satellite visibility computation
├── sample_tle.txt # Example TLE dataset for testing
├── sat_db_tle.txt # Larger TLE satellite database
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # MIT license

##Author
Upasana Panigrahi
Aerospace Engineering
University of Auckland
