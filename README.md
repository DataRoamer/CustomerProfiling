# Customer Profiling Using Clustering Analysis

This project implements customer segmentation using various clustering algorithms to identify distinct customer profiles for retail businesses.

## Overview

The customer profiling system analyzes transactional data to segment customers based on their purchasing behavior using RFM (Recency, Frequency, Monetary) analysis and additional features. It employs multiple clustering algorithms to find the optimal customer segmentation.

## Features

- **Data Loading**: Supports UCI Online Retail Dataset with fallback to synthetic data
- **Feature Engineering**: Creates comprehensive customer features including RFM metrics
- **Multiple Clustering Algorithms**: K-Means, Hierarchical Clustering, and DBSCAN
- **Cluster Optimization**: Uses silhouette analysis and elbow method to find optimal clusters
- **Visualization**: Comprehensive plots for cluster analysis and customer profiling
- **Business Insights**: Actionable customer segment profiles with marketing recommendations
- **PDF Reports**: Professional reports with theoretical framework and business intelligence

## Project Structure

```
CustomerProfiling/
├── theoretical_customer_profiling_analysis.py  # Main theoretical framework
├── enhanced_customer_profiling_with_visuals.py # Advanced visualizations
├── customer_profiling_with_pdf.py              # PDF report generation
├── customer_profiling_clustering.py            # Basic analysis
├── requirements.txt                             # Dependencies
└── README.md                                    # This file
```

## Installation

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Analysis
```bash
python customer_profiling_clustering.py
```

### Enhanced Analysis with Advanced Visualizations
```bash
python enhanced_customer_profiling_with_visuals.py
```

### Comprehensive Theoretical Analysis
```bash
python theoretical_customer_profiling_analysis.py
```

## Features

### Customer Features
1. **Recency**: Days since last purchase
2. **Frequency**: Number of transactions
3. **Monetary**: Total amount spent
4. **Product Diversity**: Number of unique products purchased
5. **Engagement Intensity**: Purchase frequency per unit time
6. **Value Velocity**: Spending rate over customer lifetime
7. **Purchase Consistency**: Behavioral stability measure

### Clustering Algorithms
- **K-Means**: Partitional clustering with optimal k selection
- **Hierarchical**: Agglomerative clustering with Ward linkage
- **DBSCAN**: Density-based clustering for outlier detection

### Statistical Validation
- **Silhouette Analysis**: Cluster quality assessment
- **ANOVA Testing**: Statistical significance of segment differences
- **Mutual Information**: Feature importance analysis
- **Ground Truth Validation**: When available

## Customer Segments

Typical segments identified:
- **VIP Customers**: High-value, frequent, recent buyers
- **Loyal Customers**: Regular, engaged customers
- **At-Risk Customers**: Haven't purchased recently
- **Low-Value Customers**: Small spenders with growth potential
- **Regular Customers**: Average engagement and spending

## Business Applications

- Targeted marketing campaigns
- Customer retention strategies
- Revenue optimization
- Resource allocation
- Personalized customer experiences

## Theoretical Framework

The analysis is grounded in established theories:
- RFM Analysis Theory (Hughes, 1994)
- Customer Lifetime Value Theory (Gupta & Lehmann, 2005)
- Market Segmentation Theory (Kotler & Keller, 2016)
- Clustering Theory (MacQueen, 1967; Ward, 1963; Ester et al., 1996)

## Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.1.0
- scipy >= 1.9.0

## License

This project is provided for educational and research purposes.