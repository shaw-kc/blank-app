"""
Santander Customer Satisfaction - Interactive Streamlit Dashboard

This dashboard provides comprehensive visualizations and insights for the Santander dataset
focusing on correlation, distribution, clustering, and target analysis.

Date: 2025
Environment: Python 3.11, Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Santander Customer Satisfaction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #EC0000;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #EC0000;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

@st.cache_data
def load_data_from_gdrive(file_id):
    """Load CSV from Google Drive"""
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(url)
    return pd.read_csv(StringIO(response.text))

def load_and_preprocess_data(uploaded_file):
    """Load the Santander dataset and perform initial preprocessing"""
    FILE_ID = "1iN88FzuSEYxsGQK52XFXJgo_VGyIhiRp"  # From share link
    df = load_data_from_gdrive(FILE_ID)
    return df

@st.cache_data
def group_columns_by_prefix(df):
    """Group columns by their prefix for organized analysis"""
    columns_dict = {
        'balances': [col for col in df.columns if col.startswith('saldo_')],
        'transaction_amounts': [col for col in df.columns if col.startswith('imp_')],
        'operation_counts': [col for col in df.columns if col.startswith('num_')],
        'indicators': [col for col in df.columns if col.startswith('ind_')],
        'changes': [col for col in df.columns if col.startswith('delta_')],
        'general': [col for col in df.columns if col.startswith('var')]
    }
    return columns_dict

@st.cache_data
def compute_correlations(df):
    """Calculate correlation with TARGET"""
    correlations = df.corr()['TARGET'].drop('TARGET').sort_values(ascending=False)
    return correlations

@st.cache_data
def perform_clustering(df, n_clusters=4):
    """Perform K-Means clustering"""
    feature_cols = df.select_dtypes(include=[np.number]).columns.drop('TARGET')
    feature_cols = feature_cols[df[feature_cols].std() > 0]
    
    if len(feature_cols) > 50:
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = variances.head(50).index
    
    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    cluster_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters,
        'TARGET': df['TARGET'].values
    })
    
    return cluster_df, clusters, pca

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def display_target_distribution(df):
    """Display target distribution with insights"""
    col1, col2 = st.columns(2)
    
    target_counts = df['TARGET'].value_counts()
    satisfied_pct = target_counts[0] / len(df) * 100
    dissatisfied_pct = target_counts[1] / len(df) * 100
    imbalance_ratio = target_counts[0] / target_counts[1]
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                x=['Satisfied (0)', 'Dissatisfied (1)'],
                y=target_counts.values,
                marker_color=[COLORS[0], COLORS[1]],
                text=[f'{v:,}<br>({v/len(df)*100:.1f}%)' for v in target_counts.values],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title='Customer Satisfaction Distribution',
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[
            go.Pie(
                labels=['Satisfied', 'Dissatisfied'],
                values=target_counts.values,
                marker_colors=[COLORS[0], COLORS[1]],
                hole=0.4
            )
        ])
        fig.update_layout(
            title='Satisfaction Rate',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Satisfied", f"{satisfied_pct:.1f}%")
    col3.metric("Dissatisfied", f"{dissatisfied_pct:.1f}%")
    col4.metric("Imbalance Ratio", f"{imbalance_ratio:.1f}:1")
    
    # Insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("### 💡 Key Insights")
    st.markdown(f"""
    - **Highly Imbalanced Dataset**: {satisfied_pct:.1f}% satisfied vs {dissatisfied_pct:.1f}% dissatisfied
    - **Class Imbalance Ratio**: {imbalance_ratio:.0f}:1 requires special handling (SMOTE, class weights)
    - **Evaluation Strategy**: Focus on Precision, Recall, F1-Score, and AUC-ROC for minority class
    - **Business Impact**: Missing dissatisfied customers (false negatives) has HIGH cost
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def display_correlation_heatmap(df, correlations, top_n=20):
    """Display correlation heatmap"""
    top_pos = correlations.head(top_n//2)
    top_neg = correlations.tail(top_n//2)
    top_features = pd.concat([top_pos, top_neg])
    
    selected_cols = list(top_features.index) + ['TARGET']
    corr_matrix = df[selected_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title=f'Correlation Heatmap - Top {top_n} Features vs TARGET',
        height=800,
        width=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Top 10 POSITIVE Correlations")
        for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
            st.text(f"{i:2d}. {feature[:35]:35s} | r = {corr:+.4f}")
    
    with col2:
        st.markdown("#### 📉 Top 10 NEGATIVE Correlations")
        for i, (feature, corr) in enumerate(correlations.tail(10).items(), 1):
            st.text(f"{i:2d}. {feature[:35]:35s} | r = {corr:+.4f}")

def display_balance_distributions(df, col_groups):
    """Display balance distributions"""
    balance_cols = col_groups['balances'][:6]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=balance_cols
    )
    
    for idx, col in enumerate(balance_cols):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1
        
        if col in df.columns:
            for target, name, color in [(0, 'Satisfied', COLORS[0]), (1, 'Dissatisfied', COLORS[1])]:
                data = df[df['TARGET'] == target][col]
                fig.add_trace(
                    go.Box(y=data, name=name, marker_color=color, showlegend=(idx == 0)),
                    row=row, col=col_pos
                )
    
    fig.update_layout(height=600, title_text="Balance Distributions: Satisfied vs Dissatisfied")
    st.plotly_chart(fig, use_container_width=True)

def display_clustering_analysis(cluster_df, pca):
    """Display clustering analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            cluster_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            title='Customer Clusters (PCA Projection)',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
            },
            color_discrete_sequence=COLORS
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            cluster_df,
            x='PC1',
            y='PC2',
            color='TARGET',
            title='Customer Satisfaction (PCA Projection)',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                'TARGET': 'Satisfaction'
            },
            color_discrete_map={0: COLORS[0], 1: COLORS[1]}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster statistics
    cluster_stats = cluster_df.groupby('Cluster')['TARGET'].agg(['mean', 'count']).reset_index()
    cluster_stats.columns = ['Cluster', 'Dissatisfaction_Rate', 'Count']
    cluster_stats['Dissatisfaction_Rate'] *= 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                x=cluster_stats['Cluster'],
                y=cluster_stats['Dissatisfaction_Rate'],
                marker_color=COLORS[:len(cluster_stats)],
                text=[f"{v:.1f}%" for v in cluster_stats['Dissatisfaction_Rate']],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title='Dissatisfaction Rate by Cluster',
            xaxis_title='Cluster',
            yaxis_title='Dissatisfaction Rate (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[
            go.Bar(
                x=cluster_stats['Cluster'],
                y=cluster_stats['Count'],
                marker_color=COLORS[:len(cluster_stats)],
                text=[f"{v:,}" for v in cluster_stats['Count']],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title='Customer Count by Cluster',
            xaxis_title='Cluster',
            yaxis_title='Number of Customers',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display cluster statistics table
    st.markdown("#### 📊 Cluster Statistics")
    overall_dissatisfaction = cluster_df['TARGET'].mean() * 100
    
    cluster_stats['% of Total'] = cluster_stats['Count'] / len(cluster_df) * 100
    cluster_stats['Risk Level'] = cluster_stats['Dissatisfaction_Rate'].apply(
        lambda x: '🔴 HIGH' if x > overall_dissatisfaction * 1.5 else 
                  '🟡 MEDIUM' if x > overall_dissatisfaction else '🟢 LOW'
    )
    
    st.dataframe(
        cluster_stats.style.format({
            'Dissatisfaction_Rate': '{:.2f}%',
            '% of Total': '{:.1f}%',
            'Count': '{:,}'
        }),
        use_container_width=True
    )

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Santander Customer Satisfaction Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload train.csv file",
        type=['csv'],
        help="Upload the Santander customer satisfaction dataset"
    )
    
    if uploaded_file is None:
        st.info("👈 Please upload the train.csv file to begin analysis")
        st.markdown("""
        ### About This Dashboard
        
        This interactive dashboard provides comprehensive exploratory data analysis (EDA) 
        for the Santander Customer Satisfaction dataset. Features include:
        
        - **Target Distribution Analysis**: Understand class imbalance and satisfaction rates
        - **Correlation Analysis**: Identify key features correlated with dissatisfaction
        - **Balance & Transaction Insights**: Compare financial patterns across customer segments
        - **Customer Activity**: Analyze operation counts and temporal changes
        - **Clustering Analysis**: Discover hidden customer segments and risk profiles
        - **Feature Importance**: Identify top predictive features
        
        Upload your dataset to get started!
        """)
        return
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_and_preprocess_data(uploaded_file)
        col_groups = group_columns_by_prefix(df)
        correlations = compute_correlations(df)
    
    # Sidebar - Dataset Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Dataset Info")
    st.sidebar.metric("Total Samples", f"{len(df):,}")
    st.sidebar.metric("Total Features", f"{len(df.columns) - 1:,}")
    st.sidebar.metric("Dissatisfaction Rate", f"{df['TARGET'].mean():.2%}")
    
    # Sidebar - Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.radio(
        "Select Analysis Section",
        [
            "📊 Overview",
            "🔗 Correlation Analysis",
            "💰 Balances & Transactions",
            "📈 Customer Activity",
            "🎯 Clustering Insights",
            "📐 Feature Importance"
        ]
    )
    
    # Sidebar - Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    if "🎯 Clustering Insights" in page:
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 4)
    
    # Main content based on selection
    if page == "📊 Overview":
        st.markdown('<h2 class="sub-header">Overview</h2>', unsafe_allow_html=True)
        
        # Dataset summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Dataset Shape", f"{df.shape[0]:,} × {df.shape[1]:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Missing Values", f"{missing_pct:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            zero_var_cols = df.columns[df.nunique() <= 1]
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Zero Variance Cols", len(zero_var_cols))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Target distribution
        st.markdown("### Target Distribution")
        display_target_distribution(df)
        
        # Column groups
        st.markdown("---")
        st.markdown("### Feature Groups")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**💵 Balances**: {len(col_groups['balances'])} columns")
            st.info(f"**💳 Transactions**: {len(col_groups['transaction_amounts'])} columns")
        
        with col2:
            st.info(f"**📊 Operations**: {len(col_groups['operation_counts'])} columns")
            st.info(f"**🔔 Indicators**: {len(col_groups['indicators'])} columns")
        
        with col3:
            st.info(f"**📉 Changes**: {len(col_groups['changes'])} columns")
            st.info(f"**🔢 General**: {len(col_groups['general'])} columns")
    
    elif page == "🔗 Correlation Analysis":
        st.markdown('<h2 class="sub-header">Correlation Analysis</h2>', unsafe_allow_html=True)
        
        top_n = st.slider("Number of top features to display", 10, 50, 20, step=5)
        
        display_correlation_heatmap(df, correlations, top_n)
        
        # Correlation statistics
        st.markdown("---")
        st.markdown("### 📊 Correlation Statistics")
        
        abs_corr = correlations.abs()
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Strongest Positive", f"{correlations.max():+.4f}")
        col2.metric("Strongest Negative", f"{correlations.min():+.4f}")
        col3.metric("Mean |r|", f"{abs_corr.mean():.4f}")
        col4.metric("Features |r| > 0.05", f"{(abs_corr > 0.05).sum()}")
    
    elif page == "💰 Balances & Transactions":
        st.markdown('<h2 class="sub-header">Balances & Transactions</h2>', unsafe_allow_html=True)
        
        st.markdown("### Balance Distributions")
        display_balance_distributions(df, col_groups)
        
        st.markdown("---")
        st.markdown("### Transaction Amount Distributions")
        
        transaction_cols = col_groups['transaction_amounts'][:6]
        
        fig = make_subplots(rows=2, cols=3, subplot_titles=transaction_cols)
        
        for idx, col in enumerate(transaction_cols):
            row = idx // 3 + 1
            col_pos = idx % 3 + 1
            
            if col in df.columns:
                for target, name, color in [(0, 'Satisfied', COLORS[0]), (1, 'Dissatisfied', COLORS[1])]:
                    data = df[df['TARGET'] == target][col]
                    fig.add_trace(
                        go.Violin(y=data, name=name, marker_color=color, showlegend=(idx == 0)),
                        row=row, col=col_pos
                    )
        
        fig.update_layout(height=600, title_text="Transaction Amount Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📈 Customer Activity":
        st.markdown('<h2 class="sub-header">Customer Activity Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("### Operation Count Distributions")
        
        operation_cols = col_groups['operation_counts'][:6]
        
        fig = make_subplots(rows=2, cols=3, subplot_titles=operation_cols)
        
        for idx, col in enumerate(operation_cols):
            row = idx // 3 + 1
            col_pos = idx % 3 + 1
            
            if col in df.columns:
                for target, name, color in [(0, 'Satisfied', COLORS[0]), (1, 'Dissatisfied', COLORS[1])]:
                    data = df[df['TARGET'] == target][col]
                    fig.add_trace(
                        go.Histogram(x=data, name=name, marker_color=color, 
                                   opacity=0.6, showlegend=(idx == 0), nbinsx=30),
                        row=row, col=col_pos
                    )
        
        fig.update_layout(height=600, title_text="Operation Count Distributions", barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
        
        # Temporal changes
        st.markdown("---")
        st.markdown("### Temporal Changes")
        
        delta_cols = [col for col in col_groups['changes'] if col in df.columns][:6]
        
        if len(delta_cols) > 0:
            fig = go.Figure()
            
            for col in delta_cols:
                avg_by_target = df.groupby('TARGET')[col].mean()
                fig.add_trace(go.Scatter(
                    x=['Satisfied', 'Dissatisfied'],
                    y=avg_by_target.values,
                    mode='lines+markers',
                    name=col,
                    line=dict(width=2),
                    marker=dict(size=10)
                ))
            
            fig.update_layout(
                title="Average Changes by Customer Satisfaction",
                xaxis_title="Customer Type",
                yaxis_title="Average Change",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🎯 Clustering Insights":
        st.markdown('<h2 class="sub-header">Clustering Insights</h2>', unsafe_allow_html=True)
        
        with st.spinner("Performing clustering analysis..."):
            cluster_df, clusters, pca = perform_clustering(df, n_clusters)
        
        display_clustering_analysis(cluster_df, pca)
    
    elif page == "📐 Feature Importance":
        st.markdown('<h2 class="sub-header">Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        top_n = st.slider("Number of top features", 5, 20, 10)
        
        with col1:
            st.markdown("### Top Features by Correlation")
            top_corr = correlations.abs().sort_values(ascending=False).head(top_n)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=top_corr.index,
                    x=top_corr.values,
                    orientation='h',
                    marker_color=COLORS[0]
                )
            ])
            fig.update_layout(
                xaxis_title='Absolute Correlation with TARGET',
                yaxis={'categoryorder': 'total ascending'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Top Features by Variance")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('TARGET')
            variances = df[numeric_cols].var().sort_values(ascending=False).head(top_n)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=variances.index,
                    x=variances.values,
                    orientation='h',
                    marker_color=COLORS[1]
                )
            ])
            fig.update_layout(
                xaxis_title='Variance',
                yaxis={'categoryorder': 'total ascending'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature comparison table
        st.markdown("---")
        st.markdown("### 📋 Feature Comparison")
        
        comparison_df = pd.DataFrame({
            'Feature': top_corr.index,
            'Abs_Correlation': top_corr.values,
            'Actual_Correlation': [correlations[f] for f in top_corr.index],
            'Direction': ['↑ Risk Factor' if correlations[f] > 0 else '↓ Protective Factor' 
                         for f in top_corr.index]
        })
        
        st.dataframe(
            comparison_df.style.format({
                'Abs_Correlation': '{:.4f}',
                'Actual_Correlation': '{:+.4f}'
            }),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
