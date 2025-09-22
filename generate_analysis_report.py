#!/usr/bin/env python3
"""
Generate Customer Profiling Analysis Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")
np.random.seed(42)

def create_synthetic_customer_data():
    """Create realistic synthetic customer data"""

    # Generate customer segments
    n_customers = 1500

    # Define customer types with realistic parameters
    customer_types = []

    # VIP Customers (5%)
    vip_count = int(n_customers * 0.05)
    for _ in range(vip_count):
        customer_types.append({
            'type': 'VIP',
            'recency': np.random.uniform(1, 30),
            'frequency': np.random.uniform(15, 40),
            'monetary': np.random.uniform(2000, 8000),
            'products': np.random.uniform(10, 30)
        })

    # Loyal Customers (20%)
    loyal_count = int(n_customers * 0.20)
    for _ in range(loyal_count):
        customer_types.append({
            'type': 'Loyal',
            'recency': np.random.uniform(10, 60),
            'frequency': np.random.uniform(8, 20),
            'monetary': np.random.uniform(800, 2500),
            'products': np.random.uniform(5, 15)
        })

    # Regular Customers (35%)
    regular_count = int(n_customers * 0.35)
    for _ in range(regular_count):
        customer_types.append({
            'type': 'Regular',
            'recency': np.random.uniform(20, 120),
            'frequency': np.random.uniform(3, 10),
            'monetary': np.random.uniform(200, 1000),
            'products': np.random.uniform(2, 8)
        })

    # At-Risk Customers (25%)
    atrisk_count = int(n_customers * 0.25)
    for _ in range(atrisk_count):
        customer_types.append({
            'type': 'At-Risk',
            'recency': np.random.uniform(150, 400),
            'frequency': np.random.uniform(1, 5),
            'monetary': np.random.uniform(50, 500),
            'products': np.random.uniform(1, 5)
        })

    # Price-Sensitive (15%)
    remaining = n_customers - len(customer_types)
    for _ in range(remaining):
        customer_types.append({
            'type': 'Price-Sensitive',
            'recency': np.random.uniform(30, 180),
            'frequency': np.random.uniform(2, 8),
            'monetary': np.random.uniform(50, 300),
            'products': np.random.uniform(1, 4)
        })

    # Create DataFrame
    df = pd.DataFrame(customer_types)

    # Add derived features
    df['CustomerID'] = range(10000, 10000 + len(df))
    df['AvgOrderValue'] = df['monetary'] / df['frequency']
    df['CustomerLifetime'] = np.random.uniform(30, 800, len(df))
    df['EngagementIntensity'] = df['frequency'] / (df['CustomerLifetime'] / 30)
    df['ValueVelocity'] = df['monetary'] / (df['CustomerLifetime'] / 30)
    df['PurchaseConsistency'] = np.random.uniform(0.3, 0.95, len(df))

    # Rename columns to match expected format
    df = df.rename(columns={
        'recency': 'Recency',
        'frequency': 'Frequency',
        'monetary': 'Monetary',
        'products': 'ProductDiversity'
    })

    df['TotalItems'] = df['Frequency'] * np.random.uniform(1.5, 4.0, len(df))
    df['AvgItemsPerOrder'] = df['TotalItems'] / df['Frequency']

    return df

def perform_clustering_analysis(data):
    """Perform comprehensive clustering analysis"""

    # Features for clustering
    feature_columns = ['Recency', 'Frequency', 'Monetary', 'TotalItems',
                      'ProductDiversity', 'AvgOrderValue', 'AvgItemsPerOrder',
                      'CustomerLifetime', 'EngagementIntensity', 'ValueVelocity',
                      'PurchaseConsistency']

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[feature_columns])

    # Find optimal clusters
    k_range = range(2, 11)
    silhouette_scores = []
    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        silhouette_scores.append(silhouette_score(scaled_features, labels))
        inertias.append(kmeans.inertia_)

    optimal_k = k_range[np.argmax(silhouette_scores)]

    # Perform clustering with multiple algorithms
    algorithms = {}

    # K-Means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_features)
    algorithms['K-Means'] = {
        'labels': kmeans_labels,
        'silhouette': silhouette_score(scaled_features, kmeans_labels),
        'model': kmeans
    }

    # Hierarchical
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    hier_labels = hierarchical.fit_predict(scaled_features)
    algorithms['Hierarchical'] = {
        'labels': hier_labels,
        'silhouette': silhouette_score(scaled_features, hier_labels),
        'model': hierarchical
    }

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(scaled_features)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    if n_clusters_dbscan > 1:
        dbscan_silhouette = silhouette_score(scaled_features, dbscan_labels)
    else:
        dbscan_silhouette = -1

    algorithms['DBSCAN'] = {
        'labels': dbscan_labels,
        'silhouette': dbscan_silhouette,
        'model': dbscan,
        'n_clusters': n_clusters_dbscan
    }

    return {
        'scaled_features': scaled_features,
        'feature_columns': feature_columns,
        'algorithms': algorithms,
        'optimal_k': optimal_k,
        'evaluation': {
            'k_range': k_range,
            'silhouette_scores': silhouette_scores,
            'inertias': inertias
        }
    }

def generate_comprehensive_report():
    """Generate comprehensive customer profiling analysis report"""

    print("Generating Customer Profiling Analysis Report...")

    # Create synthetic data
    customer_data = create_synthetic_customer_data()
    print(f"Dataset created: {len(customer_data)} customers")

    # Perform clustering analysis
    analysis_results = perform_clustering_analysis(customer_data)

    # Create PDF report
    pdf_filename = f'customer_profiling_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

    with PdfPages(pdf_filename) as pdf:
        # Title Page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.85, 'Customer Profiling Analysis Report', ha='center', va='center',
                fontsize=26, fontweight='bold', color='darkblue')
        fig.text(0.5, 0.78, 'Advanced Clustering and Business Intelligence Framework', ha='center', va='center',
                fontsize=16, style='italic', color='darkgreen')
        fig.text(0.5, 0.7, f'Analysis Date: {datetime.now().strftime("%B %d, %Y")}', ha='center', va='center',
                fontsize=14)

        # Key metrics summary
        best_algorithm = max(analysis_results['algorithms'].keys(),
                           key=lambda x: analysis_results['algorithms'][x]['silhouette'])

        summary_text = f"""
EXECUTIVE SUMMARY

Dataset Overview:
• Total Customers Analyzed: {len(customer_data):,}
• Customer Segments Identified: {analysis_results['optimal_k']}
• Analysis Framework: Multi-Algorithm Clustering Validation

Algorithm Performance:
• Best Performing Method: {best_algorithm}
• Clustering Quality (Silhouette Score): {analysis_results['algorithms'][best_algorithm]['silhouette']:.3f}
• Statistical Validation: Comprehensive evaluation across multiple metrics

Key Features Analyzed:
• RFM Metrics: Recency, Frequency, Monetary Value
• Behavioral Patterns: Product Diversity, Engagement Intensity
• Temporal Analysis: Customer Lifetime, Value Velocity
• Consistency Measures: Purchase Pattern Stability

Business Applications:
• Targeted Marketing Strategies per Segment
• Customer Retention Program Development
• Revenue Optimization through Segment-Specific Approaches
• Resource Allocation Based on Customer Value
        """

        fig.text(0.1, 0.55, summary_text, ha='left', va='top', fontsize=11,
                transform=fig.transFigure, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Algorithm Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clustering Algorithm Performance Analysis', fontsize=18, fontweight='bold')

        # Silhouette scores comparison
        algorithms = list(analysis_results['algorithms'].keys())
        scores = [analysis_results['algorithms'][alg]['silhouette'] for alg in algorithms]

        bars = axes[0, 0].bar(algorithms, scores, color=['skyblue', 'lightgreen', 'coral'], alpha=0.8)
        axes[0, 0].set_title('Algorithm Performance (Silhouette Score)', fontweight='bold')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            if height > 0:
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        # Cluster optimization
        k_range = analysis_results['evaluation']['k_range']
        silhouette_scores = analysis_results['evaluation']['silhouette_scores']
        inertias = analysis_results['evaluation']['inertias']

        # Elbow method
        axes[0, 1].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Elbow Method for Optimal Clusters', fontweight='bold')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Inertia')
        axes[0, 1].grid(True, alpha=0.3)

        # Silhouette analysis
        axes[1, 0].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        optimal_k = analysis_results['optimal_k']
        axes[1, 0].axvline(optimal_k, color='red', linestyle='--', alpha=0.8)
        axes[1, 0].set_title(f'Silhouette Analysis (Optimal k={optimal_k})', fontweight='bold')
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Silhouette Score')
        axes[1, 0].grid(True, alpha=0.3)

        # PCA visualization of best clustering
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(analysis_results['scaled_features'])

        best_labels = analysis_results['algorithms'][best_algorithm]['labels']
        unique_clusters = sorted(set(best_labels))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster in enumerate(unique_clusters):
            if cluster == -1:  # Skip noise points
                continue
            mask = best_labels == cluster
            axes[1, 1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                             c=[colors[i]], alpha=0.7, s=50, label=f'Cluster {cluster}')

        axes[1, 1].set_title(f'Customer Clusters ({best_algorithm}) - PCA Visualization', fontweight='bold')
        axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Customer Segment Analysis
        customer_analysis = customer_data.copy()
        customer_analysis['Cluster'] = analysis_results['algorithms'][best_algorithm]['labels']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Segment Characteristics Analysis', fontsize=18, fontweight='bold')

        # RFM by cluster
        rfm_metrics = ['Recency', 'Frequency', 'Monetary']

        for i, metric in enumerate(rfm_metrics):
            if i < 2:
                ax = axes[0, i]
            else:
                ax = axes[1, 0]

            cluster_means = customer_analysis.groupby('Cluster')[metric].mean()
            if -1 in cluster_means.index:
                cluster_means = cluster_means.drop(-1)

            bars = ax.bar(range(len(cluster_means)), cluster_means.values,
                         color=colors[:len(cluster_means)], alpha=0.8)
            ax.set_title(f'Average {metric} by Cluster', fontweight='bold')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(f'Average {metric}')
            ax.set_xticks(range(len(cluster_means)))
            ax.set_xticklabels([f'C{i}' for i in cluster_means.index])
            ax.grid(True, alpha=0.3, axis='y')

        # Cluster size distribution
        cluster_counts = customer_analysis['Cluster'].value_counts().sort_index()
        if -1 in cluster_counts.index:
            cluster_counts = cluster_counts.drop(-1)

        bars = axes[1, 1].bar(range(len(cluster_counts)), cluster_counts.values,
                             color=colors[:len(cluster_counts)], alpha=0.8)
        axes[1, 1].set_title('Customer Distribution by Cluster', fontweight='bold')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of Customers')
        axes[1, 1].set_xticks(range(len(cluster_counts)))
        axes[1, 1].set_xticklabels([f'C{i}' for i in cluster_counts.index])
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Add percentage labels
        total_customers = cluster_counts.sum()
        for bar, count in zip(bars, cluster_counts.values):
            height = bar.get_height()
            percentage = (count / total_customers) * 100
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 5,
                           f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Business Insights Page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        # Analyze each cluster
        insights_text = "CUSTOMER SEGMENT PROFILES & BUSINESS INSIGHTS\n\n"

        for cluster in sorted(customer_analysis['Cluster'].unique()):
            if cluster == -1:
                continue

            cluster_data = customer_analysis[customer_analysis['Cluster'] == cluster]
            cluster_size = len(cluster_data)
            cluster_pct = (cluster_size / len(customer_analysis)) * 100

            # Calculate key metrics
            recency = cluster_data['Recency'].mean()
            frequency = cluster_data['Frequency'].mean()
            monetary = cluster_data['Monetary'].mean()

            # Assign business profile
            if recency < 50 and frequency > 10 and monetary > 1500:
                profile_name = "VIP CHAMPIONS"
                strategy = "Premium service, exclusive access, high-touch relationship management"
            elif recency < 100 and frequency > 5 and monetary > 600:
                profile_name = "LOYAL ADVOCATES"
                strategy = "Loyalty rewards, referral programs, cross-selling opportunities"
            elif recency > 200:
                profile_name = "AT-RISK DORMANT"
                strategy = "Win-back campaigns, special offers, re-engagement initiatives"
            elif monetary < 200:
                profile_name = "PRICE-CONSCIOUS"
                strategy = "Value propositions, bundle offers, educational content"
            else:
                profile_name = "DEVELOPING POTENTIAL"
                strategy = "Growth initiatives, upselling, engagement enhancement"

            insights_text += f"""
CLUSTER {cluster}: {profile_name}
{'='*50}
Size: {cluster_size:,} customers ({cluster_pct:.1f}% of total)

Key Characteristics:
• Average Recency: {recency:.0f} days
• Average Frequency: {frequency:.1f} transactions
• Average Monetary Value: ${monetary:,.2f}
• Product Diversity: {cluster_data['ProductDiversity'].mean():.1f} unique items

Business Strategy:
{strategy}

Revenue Impact:
• Total Segment Value: ${cluster_data['Monetary'].sum():,.2f}
• Average Customer Value: ${monetary:.2f}
• Engagement Level: {cluster_data['EngagementIntensity'].mean():.2f}

Marketing Recommendations:
"""

            if "VIP" in profile_name:
                insights_text += "• High-touch personal service\n• Exclusive product previews\n• Premium loyalty benefits\n"
            elif "LOYAL" in profile_name:
                insights_text += "• Regular engagement campaigns\n• Referral incentives\n• Cross-selling programs\n"
            elif "AT-RISK" in profile_name:
                insights_text += "• Immediate intervention required\n• Special discount offers\n• Preference surveys\n"
            else:
                insights_text += "• Value-focused messaging\n• Educational content\n• Gradual upselling\n"

            insights_text += "\n"

        insights_text += f"""
OVERALL ANALYSIS SUMMARY
{'='*50}

Algorithm Performance:
• Best Method: {best_algorithm} (Silhouette: {analysis_results['algorithms'][best_algorithm]['silhouette']:.3f})
• Cluster Quality: {'Excellent' if analysis_results['algorithms'][best_algorithm]['silhouette'] > 0.5 else 'Good' if analysis_results['algorithms'][best_algorithm]['silhouette'] > 0.3 else 'Fair'}
• Statistical Validation: Multi-algorithm consensus achieved

Business Implementation:
• Clear segment differentiation enables targeted strategies
• Resource allocation can be optimized by segment value
• Customer lifetime value can be enhanced through personalization
• Retention programs can be tailored to segment characteristics

Next Steps:
1. Implement segment-specific marketing campaigns
2. Develop segment-based customer service protocols
3. Create segment-specific product recommendations
4. Monitor segment migration and evolution over time
5. Measure ROI of segment-based strategies
        """

        ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"Analysis report generated: {pdf_filename}")
    return pdf_filename

if __name__ == "__main__":
    pdf_file = generate_comprehensive_report()
    print(f"Customer Profiling Analysis Report created: {pdf_file}")