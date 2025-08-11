# gt_utils.py
"""
Ground Track Repeat Utilities for Satellite Constellation Analysis
Focused on repeat ground track patterns and gateway visibility
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import random
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

# Constants
MU = 398600.4418  # Earth gravitational parameter (km^3/s^2)
RE = 6378.137  # Earth radius (km)
EARTH_ROTATION_RATE = 360.9856473 / 86400  # deg/s
J2 = 1.08263e-3  # Earth's J2 perturbation coefficient

def calculate_orbital_parameters(altitude, inclination):
    """Calculate orbital parameters including J2 perturbations"""
    a = RE + altitude  # semi-major axis (km)
    n_rad = np.sqrt(MU / a**3)  # mean motion (rad/s)
    n_deg = n_rad * 180 / np.pi  # mean motion (deg/s)
    
    # Orbital period
    period_seconds = 2 * np.pi / n_rad
    period_minutes = period_seconds / 60
    
    # J2 perturbations
    inc_rad = np.radians(inclination)
    
    # Nodal precession rate (deg/day)
    omega_node = -1.5 * J2 * (RE/a)**2 * n_deg * np.cos(inc_rad) * 86400
    
    # Apsidal precession rate (deg/day)
    omega_perigee = 0.75 * J2 * (RE/a)**2 * n_deg * (5 * np.cos(inc_rad)**2 - 1) * 86400
    
    return {
        'semi_major_axis': a,
        'mean_motion_deg_s': n_deg,
        'mean_motion_rev_day': n_deg * 86400 / 360,
        'period_minutes': period_minutes,
        'nodal_precession': omega_node,
        'apsidal_precession': omega_perigee
    }

def find_repeat_ground_track_orbit(target_repeats, days, altitude_guess=833):
    """
    Find the altitude that gives a specific repeat ground track pattern
    
    Parameters:
    - target_repeats: Number of orbits in the repeat cycle
    - days: Number of days in the repeat cycle
    - altitude_guess: Initial guess for altitude (km)
    """
    
    def equation(alt):
        params = calculate_orbital_parameters(alt, 55)  # Using 55 deg inclination
        orbits_per_day = params['mean_motion_rev_day']
        actual_repeats = orbits_per_day * days
        return actual_repeats - target_repeats
    
    try:
        altitude = fsolve(equation, altitude_guess)[0]
        if altitude < 200 or altitude > 2000:
            return None
        return altitude
    except:
        return None

def calculate_ground_track_points(altitude, inclination, raan, initial_ma, duration_hours=24, time_step_minutes=1):
    """
    Calculate ground track points for a satellite
    
    Returns longitude/latitude points over time
    """
    params = calculate_orbital_parameters(altitude, inclination)
    
    # Time array
    time_steps = int(duration_hours * 60 / time_step_minutes)
    times = np.linspace(0, duration_hours * 3600, time_steps)
    
    lons = []
    lats = []
    
    for t in times:
        # Mean anomaly at time t
        ma = initial_ma + params['mean_motion_deg_s'] * t
        ma = ma % 360
        
        # RAAN at time t (with J2 precession)
        raan_t = raan + params['nodal_precession'] * (t / 86400)
        
        # True anomaly (simplified for circular orbit)
        ta = ma
        
        # Argument of latitude
        u = ta  # For circular orbit with omega = 0
        
        # Calculate latitude
        lat = np.arcsin(np.sin(np.radians(inclination)) * np.sin(np.radians(u)))
        lat = np.degrees(lat)
        
        # Calculate longitude (accounting for Earth rotation)
        lon = raan_t - EARTH_ROTATION_RATE * t
        lon = lon % 360
        if lon > 180:
            lon -= 360
            
        lons.append(lon)
        lats.append(lat)
    
    return np.array(lons), np.array(lats), times

def calculate_visibility_passes(sat_lon, sat_lat, ground_station, min_elevation=25):
    """
    Calculate visibility passes over a ground station
    
    Parameters:
    - sat_lon, sat_lat: Satellite ground track arrays
    - ground_station: dict with 'lat' and 'lon' keys
    - min_elevation: Minimum elevation angle in degrees
    """
    gs_lat = ground_station['lat']
    gs_lon = ground_station['lon']
    
    # Calculate angular distance from ground station
    angular_distances = []
    for slon, slat in zip(sat_lon, sat_lat):
        # Haversine formula
        dlat = np.radians(slat - gs_lat)
        dlon = np.radians(slon - gs_lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(gs_lat)) * np.cos(np.radians(slat)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        angular_distances.append(np.degrees(c))
    
    angular_distances = np.array(angular_distances)
    
    # Maximum angular distance for given minimum elevation
    # This is a simplified calculation
    max_angular_dist = 90 - min_elevation
    
    # Find passes (when satellite is within visibility cone)
    visible = angular_distances < max_angular_dist
    
    # Find pass segments
    passes = []
    in_pass = False
    pass_start = None
    
    for i, vis in enumerate(visible):
        if vis and not in_pass:
            in_pass = True
            pass_start = i
        elif not vis and in_pass:
            in_pass = False
            if pass_start is not None:
                passes.append((pass_start, i-1))
    
    # Handle case where pass extends to end
    if in_pass and pass_start is not None:
        passes.append((pass_start, len(visible)-1))
    
    return passes, visible

def analyze_repeat_ground_track(altitude, inclination, num_satellites, ground_station, 
                               repeat_days=1, min_elevation=25):
    """
    Analyze repeat ground track pattern for a constellation
    
    Returns detailed analysis of ground track repeats and visibility
    """
    params = calculate_orbital_parameters(altitude, inclination)
    
    # Calculate repeat pattern
    orbits_per_day = params['mean_motion_rev_day']
    orbits_in_repeat_cycle = orbits_per_day * repeat_days
    
    # Check if this gives a good repeat pattern (should be close to integer)
    repeat_quality = abs(orbits_in_repeat_cycle - round(orbits_in_repeat_cycle))
    
    # Distribute satellites
    if num_satellites == 1:
        raan_values = [0]
        ma_values = [0]
    else:
        # For repeat ground track, distribute satellites along the same ground track
        # but at different phases
        raan_values = []
        ma_values = []
        
        # Phase spacing for repeat ground track
        phase_spacing = 360 / num_satellites
        
        for i in range(num_satellites):
            # All satellites in same orbital plane for true repeat ground track
            raan_values.append(0)
            # Distribute along the orbit
            ma_values.append(i * phase_spacing)
    
    # Simulate ground tracks
    duration_hours = repeat_days * 24 + 2  # Add buffer
    all_passes = []
    visibility_data = []
    
    for i, (raan, ma) in enumerate(zip(raan_values, ma_values)):
        lons, lats, times = calculate_ground_track_points(
            altitude, inclination, raan, ma, duration_hours, time_step_minutes=0.5
        )
        
        passes, visible = calculate_visibility_passes(
            lons, lats, ground_station, min_elevation
        )
        
        all_passes.append({
            'satellite': i + 1,
            'passes': passes,
            'num_passes': len(passes),
            'lons': lons,
            'lats': lats,
            'times': times,
            'visible': visible
        })
        
        # Calculate pass durations and max elevations
        for pass_idx, (start, end) in enumerate(passes):
            duration_minutes = (times[end] - times[start]) / 60
            
            # Simple max elevation calculation
            min_dist = np.min(np.sqrt((lons[start:end+1] - ground_station['lon'])**2 + 
                                     (lats[start:end+1] - ground_station['lat'])**2))
            max_elev = 90 - min_dist  # Simplified
            
            visibility_data.append({
                'satellite': i + 1,
                'pass_number': pass_idx + 1,
                'start_time_hours': times[start] / 3600,
                'duration_minutes': duration_minutes,
                'max_elevation': max_elev
            })
    
    return {
        'orbital_params': params,
        'orbits_per_day': orbits_per_day,
        'orbits_in_repeat_cycle': orbits_in_repeat_cycle,
        'repeat_quality': repeat_quality,
        'repeat_days': repeat_days,
        'satellite_passes': all_passes,
        'visibility_summary': pd.DataFrame(visibility_data) if visibility_data else pd.DataFrame(),
        'total_passes': sum([p['num_passes'] for p in all_passes])
    }

def create_ground_track_visualization(analysis_results, ground_station, show_satellites=5):
    """
    Create comprehensive visualization of ground tracks and visibility
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1], hspace=0.3, wspace=0.3)
    
    # 1. World map with ground tracks
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot world map outline (simplified)
    ax1.plot([-180, 180, 180, -180, -180], [-90, -90, 90, 90, -90], 'k-', alpha=0.3, linewidth=0.5)
    ax1.plot([-180, 180], [0, 0], 'k--', alpha=0.2, linewidth=0.5)
    ax1.plot([0, 0], [-90, 90], 'k--', alpha=0.2, linewidth=0.5)
    
    # Plot ground tracks for first few satellites
    colors = plt.cm.tab10(np.linspace(0, 1, min(show_satellites, len(analysis_results['satellite_passes']))))
    
    for i, sat_data in enumerate(analysis_results['satellite_passes'][:show_satellites]):
        # Convert longitude to -180 to 180 range
        lons = sat_data['lons'].copy()
        lons[lons > 180] -= 360
        
        # Plot ground track
        ax1.scatter(lons[::10], sat_data['lats'][::10], c=[colors[i]], s=1, alpha=0.6, 
                   label=f'Sat {sat_data["satellite"]}')
        
        # Highlight visible portions
        if len(sat_data['visible']) > 0:
            visible_lons = lons[sat_data['visible']]
            visible_lats = sat_data['lats'][sat_data['visible']]
            ax1.scatter(visible_lons[::5], visible_lats[::5], c=[colors[i]], s=3, 
                       edgecolors='red', linewidth=0.5)
    
    # Mark ground station
    gs_lon = ground_station['lon']
    if gs_lon > 180:
        gs_lon -= 360
    ax1.plot(gs_lon, ground_station['lat'], 'r*', markersize=15, label=f"Riyadh GW")
    
    # Add visibility circle around ground station
    circle_lons = []
    circle_lats = []
    for angle in np.linspace(0, 360, 100):
        # Simplified visibility circle
        dist = 90 - 25  # For 25 degree minimum elevation
        lat = ground_station['lat'] + dist * np.sin(np.radians(angle))
        lon = gs_lon + dist * np.cos(np.radians(angle)) / np.cos(np.radians(ground_station['lat']))
        if -90 <= lat <= 90:
            circle_lats.append(lat)
            circle_lons.append(lon)
    
    if circle_lons:
        ax1.plot(circle_lons, circle_lats, 'r--', alpha=0.3, linewidth=1)
    
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    ax1.set_xlabel('Longitude (degrees)')
    ax1.set_ylabel('Latitude (degrees)')
    ax1.set_title(f'Ground Track Pattern (Repeat every {analysis_results["repeat_days"]} day(s), '
                 f'{analysis_results["orbits_in_repeat_cycle"]:.1f} orbits)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    
    # 2. Pass timeline
    ax2 = fig.add_subplot(gs[1, :])
    
    if not analysis_results['visibility_summary'].empty:
        vis_df = analysis_results['visibility_summary']
        
        # Plot passes as horizontal bars
        for idx, row in vis_df.iterrows():
            if row['satellite'] <= show_satellites:
                color_idx = row['satellite'] - 1
                ax2.barh(row['satellite'], row['duration_minutes']/60, 
                        left=row['start_time_hours'], height=0.8,
                        color=colors[color_idx], alpha=0.7)
        
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Satellite #')
        ax2.set_title(f'Visibility Timeline over Riyadh (Total: {analysis_results["total_passes"]} passes)')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, analysis_results['repeat_days'] * 24)
        
        # Add vertical lines for day boundaries
        for day in range(analysis_results['repeat_days'] + 1):
            ax2.axvline(day * 24, color='k', linestyle='--', alpha=0.3)
    
    # 3. Statistics summary
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')
    
    stats_text = f"""Orbital Parameters:
    • Altitude: {analysis_results['orbital_params']['semi_major_axis'] - RE:.1f} km
    • Period: {analysis_results['orbital_params']['period_minutes']:.1f} min
    • Orbits/day: {analysis_results['orbits_per_day']:.2f}
    • Nodal precession: {analysis_results['orbital_params']['nodal_precession']:.3f}°/day
    
    Repeat Ground Track:
    • Repeat cycle: {analysis_results['orbits_in_repeat_cycle']:.1f} orbits in {analysis_results['repeat_days']} day(s)
    • Repeat quality: {analysis_results['repeat_quality']:.4f} (closer to 0 is better)
    • Total passes visible: {analysis_results['total_passes']}"""
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')
    
    # 4. Pass duration histogram
    ax4 = fig.add_subplot(gs[2, 1])
    
    if not analysis_results['visibility_summary'].empty:
        vis_df = analysis_results['visibility_summary']
        ax4.hist(vis_df['duration_minutes'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Pass Duration (minutes)')
        ax4.set_ylabel('Number of Passes')
        ax4.set_title('Pass Duration Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        mean_duration = vis_df['duration_minutes'].mean()
        ax4.axvline(mean_duration, color='red', linestyle='--', 
                   label=f'Mean: {mean_duration:.1f} min')
        ax4.legend()
    
    plt.suptitle(f'Ground Track Repeat Analysis - {len(analysis_results["satellite_passes"])} Satellites',
                fontsize=14, fontweight='bold')
    
    return fig

def generate_tle_for_repeat_constellation(altitude, inclination, num_satellites, 
                                         repeat_orbits=14, repeat_days=1):
    """
    Generate TLE data for a repeat ground track constellation
    """
    params = calculate_orbital_parameters(altitude, inclination)
    
    # Calculate mean motion for exact repeat
    target_mean_motion = repeat_orbits / repeat_days  # rev/day
    
    # Current epoch
    current_date = datetime.now()
    epoch_year = current_date.year % 100
    day_of_year = current_date.timetuple().tm_yday
    hour_fraction = (current_date.hour + current_date.minute/60 + 
                    current_date.second/3600) / 24
    epoch_day = day_of_year + hour_fraction
    
    tle_lines = []
    
    # Generate TLEs
    for i in range(num_satellites):
        norad_id = 90000 + i
        
        # For repeat ground track, satellites are typically in the same plane
        raan = 0.0
        
        # Distribute satellites along the orbit
        mean_anomaly = (i * 360 / num_satellites) % 360
        
        # Satellite name
        sat_name = f"REPEAT_GT_{i+1:03d}"
        
        # Create TLE lines
        epoch_str = f"{epoch_year:02d}{epoch_day:012.8f}"
        
        # Line 1
        line1_body = (f"1 {norad_id:05d}U 24001A   {epoch_str} "
                     f" .00000000  00000+0  00000-0 0  9999")
        line1_body = line1_body[:68].ljust(68)
        chk1 = calculate_tle_checksum(line1_body)
        line1 = line1_body + str(chk1)
        
        # Line 2
        line2_body = (f"2 {norad_id:05d} {inclination:8.4f} {raan:8.4f} "
                     f"0000000 {0.0:8.4f} {mean_anomaly:8.4f} "
                     f"{target_mean_motion:11.8f}    1")
        line2_body = line2_body[:68].ljust(68)
        chk2 = calculate_tle_checksum(line2_body)
        line2 = line2_body + str(chk2)
        
        tle_lines.extend([sat_name, line1, line2])
    
    return tle_lines, {
        'num_satellites': num_satellites,
        'altitude': altitude,
        'inclination': inclination,
        'repeat_orbits': repeat_orbits,
        'repeat_days': repeat_days,
        'mean_motion': target_mean_motion
    }

def calculate_tle_checksum(line):
    """Calculate TLE checksum"""
    s = 0
    for char in line[:68]:
        if char.isdigit():
            s += int(char)
        elif char == '-':
            s += 1
    return s % 10

def validate_repeat_configuration(num_satellites, repeat_orbits, min_visibility_passes=1):
    """
    Validate if the configuration will provide adequate visibility
    
    Returns warnings and recommendations
    """
    warnings = []
    recommendations = []
    
    # Check if satellites are too few for the repeat pattern
    if num_satellites == 1:
        warnings.append("⚠️ Single satellite will have gaps between repeat cycles")
        recommendations.append("Consider increasing satellites to at least 3 for better coverage")
    
    if num_satellites < repeat_orbits / 2:
        warnings.append(f"⚠️ With {num_satellites} satellites and {repeat_orbits} orbit repeat, "
                       f"there may be significant coverage gaps")
        recommended_sats = max(3, repeat_orbits // 2)
        recommendations.append(f"Recommended minimum: {recommended_sats} satellites")
    
    # Check repeat pattern feasibility
    if repeat_orbits > 20:
        warnings.append("⚠️ High repeat orbit number may result in long gaps between passes")
    
    if repeat_orbits < 10:
        warnings.append("ℹ️ Low repeat orbit number provides frequent revisits")
    
    return warnings, recommendations
