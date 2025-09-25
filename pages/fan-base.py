import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Fan Specifications Dashboard",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data function
@st.cache_data
def load_fan_data():
    # Load the JSON data from the current directory
    try:
        with open('fans.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        st.error("fans.json file not found. Please make sure the file is in the same directory as this script.")
        st.info("Expected file path: fans.json")
        return pd.DataFrame()
    except json.JSONDecodeError:
        st.error("Error reading fans.json file. Please check if the JSON format is valid.")
        return pd.DataFrame()

# Main app
def main():
    st.title("🌀 Fan Specifications Dashboard")
    st.markdown("### Interactive analysis of fan performance data")
    
    # Load data
    df = load_fan_data()
    
    if df.empty:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    # Brand filter
    brands = st.sidebar.multiselect(
        "Select Brand(s)",
        options=df['brand'].unique(),
        default=df['brand'].unique()
    )
    
    # Size filter
    size_range = st.sidebar.slider(
        "Fan Size Range (mm)",
        min_value=int(df['size_mm'].min()),
        max_value=int(df['size_mm'].max()),
        value=(int(df['size_mm'].min()), int(df['size_mm'].max()))
    )
    
    # Power filter
    power_range = st.sidebar.slider(
        "Power Range (W)",
        min_value=int(df['power_w'].min()),
        max_value=int(df['power_w'].max()),
        value=(int(df['power_w'].min()), int(df['power_w'].max()))
    )
    
    # Filter data
    filtered_df = df[
        (df['brand'].isin(brands)) &
        (df['size_mm'].between(size_range[0], size_range[1])) &
        (df['power_w'].between(power_range[0], power_range[1]))
    ]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Fans", len(filtered_df))
    
    with col2:
        st.metric("Brands", filtered_df['brand'].nunique())
    
    with col3:
        avg_power = filtered_df['power_w'].mean()
        st.metric("Avg Power (W)", f"{avg_power:.1f}")
    
    with col4:
        avg_noise = filtered_df['noise_dba'].mean()
        st.metric("Avg Noise (dBA)", f"{avg_noise:.1f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Performance Charts", "📋 Data Table", "🔍 Fan Comparison"])
    
    with tab1:
        # Overview charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Flow vs Size scatter plot
            fig_scatter = px.scatter(
                filtered_df, 
                x='size_mm', 
                y='flow_free_m3h',
                color='brand',
                size='power_w',
                hover_data=['model', 'noise_dba'],
                title="Flow Rate vs Fan Size",
                labels={'size_mm': 'Fan Size (mm)', 'flow_free_m3h': 'Flow Rate (m³/h)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Power vs Noise
            fig_noise = px.scatter(
                filtered_df,
                x='power_w',
                y='noise_dba',
                color='brand',
                size='flow_free_m3h',
                hover_data=['model', 'size_mm'],
                title="Power vs Noise Level",
                labels={'power_w': 'Power (W)', 'noise_dba': 'Noise (dBA)'}
            )
            st.plotly_chart(fig_noise, use_container_width=True)
    
    with tab2:
        # Performance comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Flow comparison (free vs 50Pa)
            fig_flow = go.Figure()
            fig_flow.add_trace(go.Scatter(
                x=filtered_df['model'],
                y=filtered_df['flow_free_m3h'],
                mode='lines+markers',
                name='Free Flow',
                line=dict(color='blue')
            ))
            fig_flow.add_trace(go.Scatter(
                x=filtered_df['model'],
                y=filtered_df['flow_50pa_m3h'],
                mode='lines+markers',
                name='Flow at 50Pa',
                line=dict(color='red')
            ))
            fig_flow.update_layout(
                title="Flow Rate Comparison",
                xaxis_title="Model",
                yaxis_title="Flow Rate (m³/h)",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_flow, use_container_width=True)
        
        with col2:
            # Efficiency chart (Flow/Power ratio)
            filtered_df_copy = filtered_df.copy()
            filtered_df_copy['efficiency'] = filtered_df_copy['flow_free_m3h'] / filtered_df_copy['power_w']
            
            fig_eff = px.bar(
                filtered_df_copy,
                x='model',
                y='efficiency',
                color='brand',
                title="Fan Efficiency (Flow/Power Ratio)",
                labels={'efficiency': 'Efficiency (m³/h/W)', 'model': 'Model'}
            )
            fig_eff.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_eff, use_container_width=True)
    
    with tab3:
        # Data table
        st.subheader("Fan Specifications Table")
        
        # Add search functionality
        search_term = st.text_input("🔍 Search models:", placeholder="Enter model name or part number...")
        
        if search_term:
            filtered_df = filtered_df[filtered_df['model'].str.contains(search_term, case=False, na=False)]
        
        # Display table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "model": st.column_config.TextColumn("Model", width="medium"),
                "size_mm": st.column_config.NumberColumn("Size (mm)", width="small"),
                "flow_free_m3h": st.column_config.NumberColumn("Free Flow (m³/h)", width="small"),
                "flow_50pa_m3h": st.column_config.NumberColumn("Flow @50Pa (m³/h)", width="small"),
                "power_w": st.column_config.NumberColumn("Power (W)", width="small"),
                "noise_dba": st.column_config.NumberColumn("Noise (dBA)", width="small"),
                "mtbf_h": st.column_config.NumberColumn("MTBF (h)", width="small")
            }
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name="fan_data_filtered.csv",
            mime="text/csv"
        )
    
    with tab4:
        # Fan comparison tool
        st.subheader("Fan Comparison Tool")
        
        # Select fans to compare
        selected_fans = st.multiselect(
            "Select fans to compare (up to 5):",
            options=filtered_df['model'].tolist(),
            max_selections=5
        )
        
        if selected_fans:
            comparison_df = filtered_df[filtered_df['model'].isin(selected_fans)]
            
            # Create comparison radar chart
            categories = ['flow_free_m3h', 'power_w', 'noise_dba', 'size_mm']
            category_labels = ['Flow Rate', 'Power', 'Noise', 'Size']
            
            fig_radar = go.Figure()
            
            for idx, (_, row) in enumerate(comparison_df.iterrows()):
                values = []
                for cat in categories:
                    # Normalize values to 0-100 scale for radar chart
                    max_val = df[cat].max()
                    min_val = df[cat].min()
                    if cat == 'noise_dba':  # For noise, lower is better, so invert
                        normalized = 100 - ((row[cat] - min_val) / (max_val - min_val) * 100)
                    else:
                        normalized = (row[cat] - min_val) / (max_val - min_val) * 100
                    values.append(normalized)
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=category_labels + [category_labels[0]],
                    fill='toself',
                    name=row['model'],
                    opacity=0.6
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Fan Performance Comparison (Normalized)",
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Comparison table
            st.subheader("Side-by-Side Comparison")
            comparison_display = comparison_df.set_index('model').transpose()
            st.dataframe(comparison_display, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created with Streamlit • Data includes Rittal and Schrack fan specifications*")

if __name__ == "__main__":
    main()