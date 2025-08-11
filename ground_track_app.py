# streamlit_ground_track.py
"""
Ground Track Repeat Visualization and Analysis Tool
For satellite constellation planning with focus on gateway visibility
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from gt_utils import (
    calculate_orbital_parameters,
    find_repeat_ground_track_orbit,
    analyze_repeat_ground_track,
    create_ground_track_visualization,
    generate_tle_for_repeat_constellation,
    validate_repeat_configuration
)

# Page configuration
st.set_page_config(
    page_title="Ground Track Repeat Analyzer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🛰️ Ground Track Repeat Analyzer")
    st.markdown("Analyze satellite constellation repeat ground tracks and gateway visibility")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Configuration", "📊 Analysis", "📚 Help"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Constellation Parameters")
            
            # Basic parameters
            num_satellites = st.number_input(
                "Number of Satellites",
                min_value=1,
                max_value=1190,
                value=250,
                step=1,
                help="Total number of satellites in constellation (1-1190)"
            )
            
            altitude = st.number_input(
                "Altitude (km)",
                min_value=200,
                max_value=2000,
                value=833,
                step=1,
                help="Orbital altitude above Earth's surface"
            )
            
            inclination = st.number_input(
                "Inclination (degrees)",
                min_value=0.0,
                max_value=180.0,
                value=55.0,
                step=0.1,
                help="Orbital inclination angle"
            )
            
            st.markdown("---")
            st.subheader("Ground Track Repeat")
            
            repeat_mode = st.radio(
                "Configuration Mode",
                ["Standard (14 orbits/day)", "Custom"],
                help="Choose standard 14-orbit repeat or custom configuration"
            )
            
            if repeat_mode == "Standard (14 orbits/day)":
                repeat_orbits = 14
                repeat_days = 1
                st.info("📍 Standard configuration: 14 orbits in 1 day")
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    repeat_orbits = st.number_input(
                        "Repeat Orbits",
                        min_value=1,
                        max_value=30,
                        value=14,
                        help="Number of orbits in repeat cycle"
                    )
                with col_b:
                    repeat_days = st.number_input(
                        "Repeat Days",
                        min_value=1,
                        max_value=7,
                        value=1,
                        help="Number of days in repeat cycle"
                    )
            
            st.markdown("---")
            st.subheader("Ground Station")
            
            # Riyadh coordinates
            use_riyadh = st.checkbox("Use Riyadh Gateway", value=True)
            
            if use_riyadh:
                gs_lat = 24.7136
                gs_lon = 46.6753
                st.info("📍 Riyadh, Saudi Arabia")
            else:
                gs_lat = st.number_input("Gateway Latitude", value=24.7136, 
                                        min_value=-90.0, max_value=90.0)
                gs_lon = st.number_input("Gateway Longitude", value=46.6753,
                                        min_value=-180.0, max_value=180.0)
            
            ground_station = {'lat': gs_lat, 'lon': gs_lon, 'name': 'Riyadh' if use_riyadh else 'Custom'}
            
            st.markdown("---")
            st.subheader("Elevation Constraints")
            
            min_elevation_user = st.slider(
                "Min Elevation - User Terminal (°)",
                min_value=5,
                max_value=45,
                value=25,
                help="Minimum elevation angle for user terminals"
            )
            
            min_elevation_gateway = st.slider(
                "Min Elevation - Gateway (°)",
                min_value=5,
                max_value=45,
                value=25,
                help="Minimum elevation angle for gateway station"
            )
        
        with col2:
            st.subheader("Configuration Validation")
            
            # Validate configuration
            warnings, recommendations = validate_repeat_configuration(
                num_satellites, repeat_orbits
            )
            
            if warnings:
                st.warning("**Configuration Warnings:**")
                for warning in warnings:
                    st.write(warning)
            
            if recommendations:
                st.info("**Recommendations:**")
                for rec in recommendations:
                    st.write(rec)
            
            # Calculate orbital parameters
            params = calculate_orbital_parameters(altitude, inclination)
            
            st.markdown("---")
            st.subheader("Orbital Characteristics")
            
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.metric("Orbital Period", f"{params['period_minutes']:.1f} min")
                st.metric("Mean Motion", f"{params['mean_motion_rev_day']:.2f} rev/day")
            with col_2:
                st.metric("Semi-major Axis", f"{params['semi_major_axis']:.1f} km")
                st.metric("Velocity", f"{np.sqrt(MU/params['semi_major_axis']):.2f} km/s")
            with col_3:
                st.metric("Nodal Precession", f"{params['nodal_precession']:.3f}°/day")
                coverage_radius = RE * np.arccos(RE/(RE + altitude)) * 180/np.pi
                st.metric("Coverage Radius", f"{coverage_radius:.1f}°")
            
            # Suggest optimal altitude for exact repeat
            if st.checkbox("Calculate optimal altitude for exact repeat"):
                optimal_alt = find_repeat_ground_track_orbit(repeat_orbits, repeat_days, altitude)
                if optimal_alt:
                    st.success(f"✅ Optimal altitude for {repeat_orbits}/{repeat_days} repeat: "
                             f"**{optimal_alt:.2f} km**")
                    if abs(optimal_alt - altitude) > 1:
                        st.info(f"Current altitude ({altitude} km) differs by "
                               f"{abs(optimal_alt - altitude):.1f} km")
                else:
                    st.error("Could not find valid altitude for this repeat pattern")
    
    with tab2:
        st.subheader("Ground Track Analysis")
        
        if st.button("🔍 Analyze Ground Track Pattern", type="primary"):
            with st.spinner("Analyzing ground track pattern and visibility..."):
                
                # Run analysis
                analysis = analyze_repeat_ground_track(
                    altitude, inclination, num_satellites,
                    ground_station, repeat_days, min_elevation_gateway
                )
                
                # Store in session state
                st.session_state.analysis = analysis
                st.session_state.ground_station = ground_station
                st.session_state.params = {
                    'altitude': altitude,
                    'inclination': inclination,
                    'num_satellites': num_satellites,
                    'repeat_orbits': repeat_orbits,
                    'repeat_days': repeat_days
                }
        
        # Display results if analysis exists
        if 'analysis' in st.session_state:
            analysis = st.session_state.analysis
            ground_station = st.session_state.ground_station
            params = st.session_state.params
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Passes", analysis['total_passes'])
            with col2:
                st.metric("Repeat Quality", f"{analysis['repeat_quality']:.5f}",
                         help="Closer to 0 is better")
            with col3:
                st.metric("Orbits/Day", f"{analysis['orbits_per_day']:.2f}")
            with col4:
                if not analysis['visibility_summary'].empty:
                    avg_duration = analysis['visibility_summary']['duration_minutes'].mean()
                    st.metric("Avg Pass Duration", f"{avg_duration:.1f} min")
                else:
                    st.metric("Avg Pass Duration", "No passes")
            
            # Visualization
            st.markdown("---")
            
            # Control how many satellites to show
            max_sats_to_show = min(10, params['num_satellites'])
            show_sats = st.slider(
                "Satellites to visualize",
                min_value=1,
                max_value=max_sats_to_show,
                value=min(5, params['num_satellites']),
                help="Number of satellites to show in visualization (max 10)"
            )
            
            # Create visualization
            fig = create_ground_track_visualization(
                analysis, ground_station, show_satellites=show_sats
            )
            st.pyplot(fig)
            
            # Download visualization
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="📥 Download Visualization (PNG)",
                data=buf.getvalue(),
                file_name=f"ground_track_{params['num_satellites']}sats_{params['altitude']}km.png",
                mime="image/png"
            )
            
            # Detailed pass information
            st.markdown("---")
            st.subheader("Pass Details")
            
            if not analysis['visibility_summary'].empty:
                vis_df = analysis['visibility_summary']
                
                # Summary statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pass Statistics:**")
                    st.write(f"- Total passes in {repeat_days} day(s): {len(vis_df)}")
                    st.write(f"- Average passes per satellite: {len(vis_df)/params['num_satellites']:.1f}")
                    st.write(f"- Shortest pass: {vis_df['duration_minutes'].min():.1f} min")
                    st.write(f"- Longest pass: {vis_df['duration_minutes'].max():.1f} min")
                
                with col2:
                    st.markdown("**Coverage Analysis:**")
                    total_coverage_time = vis_df['duration_minutes'].sum()
                    total_time = repeat_days * 24 * 60
                    coverage_percent = (total_coverage_time / total_time) * 100
                    st.write(f"- Total coverage time: {total_coverage_time:.1f} min")
                    st.write(f"- Coverage percentage: {coverage_percent:.1f}%")
                    st.write(f"- Max elevation achieved: {vis_df['max_elevation'].max():.1f}°")
                    st.write(f"- Average max elevation: {vis_df['max_elevation'].mean():.1f}°")
                
                # Show pass table
                st.markdown("**Detailed Pass Schedule:**")
                
                # Filter by satellite if many
                if params['num_satellites'] > 10:
                    selected_sats = st.multiselect(
                        "Filter by satellite:",
                        options=list(range(1, min(show_sats + 1, params['num_satellites'] + 1))),
                        default=list(range(1, min(6, params['num_satellites'] + 1)))
                    )
                    if selected_sats:
                        filtered_df = vis_df[vis_df['satellite'].isin(selected_sats)]
                    else:
                        filtered_df = vis_df
                else:
                    filtered_df = vis_df
                
                # Format the dataframe for display
                display_df = filtered_df.copy()
                display_df['start_time'] = pd.to_datetime(display_df['start_time_hours'] * 3600, unit='s').dt.strftime('%H:%M:%S')
                display_df['duration'] = display_df['duration_minutes'].round(1).astype(str) + ' min'
                display_df['max_elev'] = display_df['max_elevation'].round(1).astype(str) + '°'
                
                st.dataframe(
                    display_df[['satellite', 'pass_number', 'start_time', 'duration', 'max_elev']],
                    use_container_width=True
                )
                
                # Download pass data
                csv_data = vis_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Pass Data (CSV)",
                    data=csv_data,
                    file_name=f"pass_schedule_{params['num_satellites']}sats.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No passes detected with current configuration")
            
            # TLE Generation
            st.markdown("---")
            st.subheader("TLE Generation")
            
            if st.button("Generate TLE File"):
                tle_lines, tle_info = generate_tle_for_repeat_constellation(
                    params['altitude'], 
                    params['inclination'],
                    params['num_satellites'],
                    params['repeat_orbits'],
                    params['repeat_days']
                )
                
                # Show preview
                st.markdown("**TLE Preview (first 3 satellites):**")
                preview = '\n'.join(tle_lines[:9])  # 3 satellites * 3 lines
                st.code(preview, language='text')
                
                # Download TLE
                tle_content = '\r\n'.join(tle_lines) + '\r\n'
                st.download_button(
                    label="📥 Download Complete TLE File",
                    data=tle_content,
                    file_name=f"repeat_gt_{params['num_satellites']}sats_{params['repeat_orbits']}orbits.tle",
                    mime="text/plain"
                )
                
                st.success(f"✅ Generated TLE for {params['num_satellites']} satellites")
    
    with tab3:
        st.subheader("📚 Understanding Repeat Ground Tracks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### What is a Repeat Ground Track?
            
            A **repeat ground track** occurs when a satellite returns to the same position 
            relative to Earth's surface after a specific number of orbits. This creates a 
            predictable pattern of coverage.
            
            #### Key Parameters:
            - **Repeat Orbits**: Number of orbits to complete the pattern
            - **Repeat Days**: Time period for the pattern to repeat
            - **Example**: 14 orbits/day means the satellite passes over the same ground 
              locations every 24 hours
            
            ### Why Use Repeat Ground Tracks?
            
            1. **Predictable Coverage**: Know exactly when satellites will be visible
            2. **Regular Service**: Consistent communication windows
            3. **Mission Planning**: Schedule operations based on known pass times
            4. **Resource Optimization**: Efficient use of ground stations
            """)
            
            st.info("""
            **💡 Tip**: For gateway stations, repeat ground tracks ensure regular 
            communication windows, critical for:
            - Data downloads
            - Command uploads  
            - Network synchronization
            - Service continuity
            """)
        
        with col2:
            st.markdown("""
            ### Configuration Guidelines
            
            #### Number of Satellites:
            - **Minimum**: 3-5 satellites for basic coverage
            - **Recommended**: 10+ for reduced gaps
            - **Optimal**: Match satellites to repeat orbits (e.g., 14 satellites for 14 orbits/day)
            
            #### Altitude Selection:
            - **LEO (200-600 km)**: Short passes, frequent revisits
            - **MEO (600-1200 km)**: Longer passes, wider coverage
            - **Higher (1200-2000 km)**: Maximum coverage, fewer passes
            
            #### Inclination Impact:
            - **0° (Equatorial)**: Coverage limited to equatorial regions
            - **55° (Typical)**: Good global coverage, suitable for most locations
            - **90° (Polar)**: Complete global coverage including poles
            - **Sun-synchronous (~98°)**: Special case for Earth observation
            
            ### Warning Indicators
            
            ⚠️ **Coverage Gaps**: Too few satellites for the repeat pattern
            
            ⚠️ **Long Gaps**: High repeat orbit number with few satellites
            
            ℹ️ **Optimization**: Consider altitude adjustment for exact repeat
            """)
        
        st.markdown("---")
        st.subheader("🎯 Riyadh Gateway Specifics")
        
        st.markdown("""
        **Location**: 24.7136°N, 46.6753°E
        
        **Optimal Parameters for Riyadh:**
        - Inclination: 45-65° for good visibility
        - Minimum elevation: 25° balances coverage and link quality
        - Repeat pattern: 14-15 orbits/day provides regular passes
        
        **Expected Performance:**
        - Pass duration: 5-8 minutes (typical at 833 km)
        - Daily passes: Depends on constellation size
        - Coverage gaps: Minimize with proper phasing
        """)
        
        # Example configurations
        st.markdown("---")
        st.subheader("📋 Example Configurations")
        
        examples = pd.DataFrame({
            'Configuration': ['Small Demo', 'Basic Coverage', 'Enhanced Service', 'Full Constellation'],
            'Satellites': [10, 50, 250, 1190],
            'Altitude (km)': [500, 650, 833, 833],
            'Inclination (°)': [45, 55, 55, 55],
            'Repeat Pattern': ['15/1', '14/1', '14/1', '14/1'],
            'Daily Passes': ['~20', '~100', '~500', '~2380'],
            'Gap Duration': ['~70 min', '~14 min', '~3 min', '<1 min']
        })
        
        st.dataframe(examples, use_container_width=True)
        
        st.markdown("---")
        st.info("""
        **🔧 Technical Notes:**
        - TLE files use Windows line endings (CR+LF) for NCAT compatibility
        - NORAD IDs are randomized to avoid conflicts
        - J2 perturbation effects are included in calculations
        - Simplified elevation model used for quick analysis
        """)

# Import constants from gt_utils
from gt_utils import MU, RE

if __name__ == "__main__":
    main()