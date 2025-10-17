import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="M3 Insurance Analytics", page_icon="🏥", layout="wide")

# Title and description
st.title("🏥 M3 Insurance Analytics Dashboard")
st.markdown("**Interactive analysis of insurance data for data-driven insights**")

# File uploader
uploaded_file = st.file_uploader("Upload your insurance dataset (CSV)", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Display basic info
    st.success(f"✅ Dataset loaded successfully! {len(df)} records found.")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Deep Dive", "🗺️ Regional Analysis", "💡 Insights"])
    
    # TAB 1: Overview
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            st.metric("Average Charges", f"${df['charges'].mean():,.0f}")
        with col3:
            st.metric("Average Age", f"{df['age'].mean():.1f} years")
        with col4:
            st.metric("Average BMI", f"{df['bmi'].mean():.1f}")
        
        st.markdown("---")
        
        # Two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Charges Distribution")
            fig1 = px.histogram(df, x='charges', nbins=50, 
                               title="Distribution of Insurance Charges",
                               labels={'charges': 'Insurance Charges ($)'},
                               color_discrete_sequence=['#1f77b4'])
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Age Distribution")
            fig2 = px.histogram(df, x='age', nbins=30,
                               title="Customer Age Distribution",
                               labels={'age': 'Age (years)'},
                               color_discrete_sequence=['#2ca02c'])
            fig2.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Demographics breakdown
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Gender Split")
            gender_counts = df['sex'].value_counts()
            fig3 = px.pie(values=gender_counts.values, names=gender_counts.index,
                         title="Customer Gender Distribution",
                         color_discrete_sequence=['#ff7f0e', '#1f77b4'])
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("Smoker Status")
            smoker_counts = df['smoker'].value_counts()
            fig4 = px.pie(values=smoker_counts.values, names=smoker_counts.index,
                         title="Smoking Status Distribution",
                         color_discrete_sequence=['#d62728', '#2ca02c'])
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, use_container_width=True)
        
        with col3:
            st.subheader("Children Distribution")
            children_counts = df['children'].value_counts().sort_index()
            fig5 = px.bar(x=children_counts.index, y=children_counts.values,
                         title="Number of Children",
                         labels={'x': 'Number of Children', 'y': 'Count'},
                         color_discrete_sequence=['#9467bd'])
            fig5.update_layout(height=350)
            st.plotly_chart(fig5, use_container_width=True)
    
    # TAB 2: Deep Dive
    with tab2:
        st.subheader("🔍 Interactive Analysis")
        
        # Interactive filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_range = st.slider("Select Age Range", 
                                 int(df['age'].min()), 
                                 int(df['age'].max()), 
                                 (int(df['age'].min()), int(df['age'].max())))
        
        with col2:
            selected_smoker = st.multiselect("Smoker Status", 
                                            options=df['smoker'].unique().tolist(),
                                            default=df['smoker'].unique().tolist())
        
        with col3:
            selected_sex = st.multiselect("Gender", 
                                         options=df['sex'].unique().tolist(),
                                         default=df['sex'].unique().tolist())
        
        # Filter data
        filtered_df = df[
            (df['age'] >= age_range[0]) & 
            (df['age'] <= age_range[1]) &
            (df['smoker'].isin(selected_smoker)) &
            (df['sex'].isin(selected_sex))
        ]
        
        st.info(f"Showing {len(filtered_df)} out of {len(df)} records")
        
        # Scatter plot: BMI vs Charges
        st.subheader("BMI vs Insurance Charges")
        fig6 = px.scatter(filtered_df, x='bmi', y='charges', 
                         color='smoker', size='age',
                         hover_data=['age', 'sex', 'children', 'region'],
                         title="Relationship between BMI and Insurance Charges",
                         labels={'bmi': 'Body Mass Index (BMI)', 
                                'charges': 'Insurance Charges ($)'},
                         color_discrete_map={'yes': '#d62728', 'no': '#2ca02c'})
        fig6.update_layout(height=500)
        st.plotly_chart(fig6, use_container_width=True)
        
        # Age vs Charges
        st.subheader("Age vs Insurance Charges")
        fig7 = px.scatter(filtered_df, x='age', y='charges', 
                         color='smoker', size='bmi',
                         hover_data=['bmi', 'sex', 'children', 'region'],
                         title="How Age Affects Insurance Charges",
                         labels={'age': 'Age (years)', 
                                'charges': 'Insurance Charges ($)'},
                         color_discrete_map={'yes': '#d62728', 'no': '#2ca02c'},
                         trendline="ols")
        fig7.update_layout(height=500)
        st.plotly_chart(fig7, use_container_width=True)
        
        # Box plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Charges by Smoker Status")
            fig8 = px.box(filtered_df, x='smoker', y='charges',
                         color='smoker',
                         title="Charges Distribution by Smoking Status",
                         labels={'smoker': 'Smoker Status', 
                                'charges': 'Insurance Charges ($)'},
                         color_discrete_map={'yes': '#d62728', 'no': '#2ca02c'})
            fig8.update_layout(height=400)
            st.plotly_chart(fig8, use_container_width=True)
        
        with col2:
            st.subheader("Charges by Gender")
            fig9 = px.box(filtered_df, x='sex', y='charges',
                         color='sex',
                         title="Charges Distribution by Gender",
                         labels={'sex': 'Gender', 
                                'charges': 'Insurance Charges ($)'},
                         color_discrete_map={'male': '#1f77b4', 'female': '#ff7f0e'})
            fig9.update_layout(height=400)
            st.plotly_chart(fig9, use_container_width=True)
    
    # TAB 3: Regional Analysis
    with tab3:
        st.subheader("🗺️ Regional Insurance Analysis")
        
        # Regional statistics
        regional_stats = df.groupby('region').agg({
            'charges': ['mean', 'count'],
            'age': 'mean',
            'bmi': 'mean'
        }).round(2)
        regional_stats.columns = ['Avg Charges', 'Count', 'Avg Age', 'Avg BMI']
        regional_stats = regional_stats.reset_index()
        
        # Region coordinates (approximate US regions)
        region_coords = {
            'southwest': {'lat': 34.5, 'lon': -111.0},
            'southeast': {'lat': 33.5, 'lon': -82.0},
            'northwest': {'lat': 45.5, 'lon': -122.0},
            'northeast': {'lat': 42.5, 'lon': -75.0}
        }
        
        # Add coordinates
        regional_stats['lat'] = regional_stats['region'].map(lambda x: region_coords[x]['lat'])
        regional_stats['lon'] = regional_stats['region'].map(lambda x: region_coords[x]['lon'])
        
        # Interactive map
        fig10 = px.scatter_geo(regional_stats,
                              lat='lat',
                              lon='lon',
                              size='Count',
                              color='Avg Charges',
                              hover_name='region',
                              hover_data={'Avg Charges': ':,.0f',
                                        'Count': True,
                                        'Avg Age': ':.1f',
                                        'Avg BMI': ':.1f',
                                        'lat': False,
                                        'lon': False},
                              title='Insurance Charges by US Region',
                              scope='usa',
                              color_continuous_scale='Viridis',
                              size_max=50)
        fig10.update_layout(height=500)
        st.plotly_chart(fig10, use_container_width=True)
        
        # Regional breakdown charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Charges by Region")
            fig11 = px.bar(regional_stats.sort_values('Avg Charges', ascending=False),
                          x='region', y='Avg Charges',
                          title="Average Insurance Charges by Region",
                          labels={'region': 'Region', 'Avg Charges': 'Average Charges ($)'},
                          color='Avg Charges',
                          color_continuous_scale='Blues')
            fig11.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig11, use_container_width=True)
        
        with col2:
            st.subheader("Customer Distribution by Region")
            fig12 = px.pie(regional_stats, values='Count', names='region',
                          title="Customer Count by Region",
                          color_discrete_sequence=px.colors.qualitative.Set3)
            fig12.update_layout(height=400)
            st.plotly_chart(fig12, use_container_width=True)
        
        # Detailed regional table
        st.subheader("Regional Statistics Summary")
        st.dataframe(regional_stats[['region', 'Avg Charges', 'Count', 'Avg Age', 'Avg BMI']]
                    .style.format({'Avg Charges': '${:,.2f}', 
                                  'Avg Age': '{:.1f}',
                                  'Avg BMI': '{:.2f}'}),
                    use_container_width=True)
    
    # TAB 4: Key Insights
    with tab4:
        st.subheader("💡 Key Insights & Recommendations")
        
        # Calculate insights
        smoker_avg = df[df['smoker'] == 'yes']['charges'].mean()
        non_smoker_avg = df[df['smoker'] == 'no']['charges'].mean()
        smoker_multiplier = smoker_avg / non_smoker_avg
        
        high_bmi = df[df['bmi'] > 30]['charges'].mean()
        normal_bmi = df[df['bmi'] <= 30]['charges'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚬 Smoking Impact")
            st.metric("Smoker Avg Charges", f"${smoker_avg:,.0f}")
            st.metric("Non-Smoker Avg Charges", f"${non_smoker_avg:,.0f}")
            st.metric("Difference", f"{smoker_multiplier:.1f}x higher", 
                     delta=f"${smoker_avg - non_smoker_avg:,.0f}")
            st.info("**Insight:** Smokers pay significantly more for insurance, representing a major risk factor.")
        
        with col2:
            st.markdown("### ⚖️ BMI Impact")
            st.metric("High BMI Avg Charges (>30)", f"${high_bmi:,.0f}")
            st.metric("Normal BMI Avg Charges (≤30)", f"${normal_bmi:,.0f}")
            st.metric("Difference", f"${high_bmi - normal_bmi:,.0f}")
            st.info("**Insight:** Higher BMI correlates with increased insurance charges.")
        
        st.markdown("---")
        
        st.markdown("### 📈 Business Recommendations")
        
        st.markdown("""
        **1. Risk-Based Pricing Strategy**
        - Implement tiered pricing based on smoking status and BMI categories
        - Smokers represent the highest cost segment - consider wellness programs
        
        **2. Preventive Care Initiatives**
        - Offer smoking cessation programs with premium discounts
        - Promote healthy lifestyle initiatives to reduce BMI-related costs
        
        **3. Regional Focus**
        - Analyze regional differences to tailor marketing and pricing strategies
        - Consider regional health trends and demographics
        
        **4. Age-Based Products**
        - Develop specialized products for different age segments
        - Consider family plans for customers with children
        
        **5. Data-Driven Underwriting**
        - Use predictive models to better assess risk
        - Continuously monitor and update pricing models based on claims data
        """)
        
        # Correlation heatmap
        st.markdown("### 🔗 Factor Correlations")
        numeric_df = df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()
        
        fig13 = px.imshow(corr_matrix,
                         labels=dict(color="Correlation"),
                         x=corr_matrix.columns,
                         y=corr_matrix.columns,
                         color_continuous_scale='RdBu_r',
                         title="Correlation Matrix of Numeric Variables")
        fig13.update_layout(height=500)
        st.plotly_chart(fig13, use_container_width=True)

else:
    st.info("👆 Please upload your insurance dataset CSV file to begin the analysis.")
    st.markdown("""
    ### Expected Dataset Format:
    - **age**: Customer age
    - **sex**: Gender (male/female)
    - **bmi**: Body Mass Index
    - **children**: Number of children/dependents
    - **smoker**: Smoking status (yes/no)
    - **region**: Geographic region
    - **charges**: Insurance charges
    """)
