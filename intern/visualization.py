import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def setup_style():
    """Set up plotting style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12

def plot_survival_rate(df):
    """Plot overall survival rate"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    survival_counts = df['Survived'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4']
    ax[0].bar(['Did Not Survive', 'Survived'], survival_counts.values, color=colors)
    ax[0].set_title('Survival Count', fontsize=14, fontweight='bold')
    ax[0].set_ylabel('Number of Passengers')
    
    # Pie chart
    survival_rate = df['Survived'].mean() * 100
    ax[1].pie(survival_counts.values, labels=['Did Not Survive', 'Survived'], 
              autopct='%1.1f%%', colors=colors, startangle=90)
    ax[1].set_title(f'Survival Rate: {survival_rate:.1f}%', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_survival_by_gender(df):
    """Plot survival rate by gender"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    survival_by_gender = pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100
    survival_by_gender.columns = ['Did Not Survive', 'Survived']
    
    survival_by_gender.plot(kind='bar', ax=ax, color=['#ff6b6b', '#4ecdc4'])
    ax.set_title('Survival Rate by Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Status')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Add value labels
    for i, (idx, row) in enumerate(survival_by_gender.iterrows()):
        ax.text(i, row['Survived'] - 5, f"{row['Survived']:.1f}%", 
                ha='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    return fig

def plot_survival_by_class(df):
    """Plot survival rate by passenger class"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    survival_by_class = pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100
    survival_by_class.columns = ['Did Not Survive', 'Survived']
    
    survival_by_class.plot(kind='bar', ax=ax, color=['#ff6b6b', '#4ecdc4'])
    ax.set_title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
    ax.set_xlabel('Passenger Class')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticklabels(['First Class', 'Second Class', 'Third Class'], rotation=0)
    ax.legend(title='Status')
    
    # Add value labels
    for i, (idx, row) in enumerate(survival_by_class.iterrows()):
        ax.text(i, row['Survived'] - 5, f"{row['Survived']:.1f}%", 
                ha='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    return fig

def plot_age_distribution(df):
    """Plot age distribution and survival"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Age histogram
    axes[0].hist(df[df['Survived'] == 0]['Age'].dropna(), bins=30, 
                 alpha=0.6, label='Did Not Survive', color='#ff6b6b')
    axes[0].hist(df[df['Survived'] == 1]['Age'].dropna(), bins=30, 
                 alpha=0.6, label='Survived', color='#4ecdc4')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Age Distribution by Survival Status')
    axes[0].legend()
    
    # Survival rate by age group
    if 'AgeGroup' in df.columns:
        survival_by_age = df.groupby('AgeGroup')['Survived'].mean() * 100
        axes[1].bar(range(len(survival_by_age)), survival_by_age.values, color='#4ecdc4')
        axes[1].set_xticks(range(len(survival_by_age)))
        axes[1].set_xticklabels(survival_by_age.index, rotation=45)
        axes[1].set_ylabel('Survival Rate (%)')
        axes[1].set_title('Survival Rate by Age Group')
        axes[1].axhline(y=df['Survived'].mean() * 100, color='red', 
                       linestyle='--', label='Overall Average')
        axes[1].legend()
        
        # Add value labels
        for i, v in enumerate(survival_by_age.values):
            axes[1].text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    return fig

def plot_fare_analysis(df):
    """Plot fare distribution and survival"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    df.boxplot(column='Fare', by='Survived', ax=axes[0])
    axes[0].set_title('Fare Distribution by Survival')
    axes[0].set_xlabel('Survived')
    axes[0].set_ylabel('Fare ($)')
    
    # Fare groups
    df['FareGroup'] = pd.qcut(df['Fare'], 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    survival_by_fare = df.groupby('FareGroup')['Survived'].mean() * 100
    
    axes[1].bar(range(len(survival_by_fare)), survival_by_fare.values, color='#4ecdc4')
    axes[1].set_xticks(range(len(survival_by_fare)))
    axes[1].set_xticklabels(survival_by_fare.index)
    axes[1].set_ylabel('Survival Rate (%)')
    axes[1].set_title('Survival Rate by Fare Group')
    
    for i, v in enumerate(survival_by_fare.values):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    """Plot correlation heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select numeric columns for correlation
    numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
    correlation_matrix = df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_family_impact(df):
    """Plot family size impact on survival"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Survival rate by family size
    survival_by_family = df.groupby('FamilySize')['Survived'].mean() * 100
    
    ax1.bar(survival_by_family.index, survival_by_family.values, color='#4ecdc4')
    ax1.set_xlabel('Family Size')
    ax1.set_ylabel('Survival Rate (%)')
    ax1.set_title('Survival Rate by Family Size')
    ax1.axhline(y=df['Survived'].mean() * 100, color='red', 
               linestyle='--', label='Overall Average')
    ax1.legend()
    
    # Family size distribution
    family_counts = df['FamilySize'].value_counts().sort_index()
    ax2.bar(family_counts.index, family_counts.values, color='#95a5a6')
    ax2.set_xlabel('Family Size')
    ax2.set_ylabel('Number of Passengers')
    ax2.set_title('Family Size Distribution')
    
    plt.tight_layout()
    return fig

def plot_embarkation_port(df):
    """Plot embarkation port analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Survival by embarkation port
    survival_by_port = df.groupby('Embarked')['Survived'].mean() * 100
    
    axes[0].bar(range(len(survival_by_port)), survival_by_port.values, color='#4ecdc4')
    axes[0].set_xticks(range(len(survival_by_port)))
    axes[0].set_xticklabels(survival_by_port.index)
    axes[0].set_ylabel('Survival Rate (%)')
    axes[0].set_title('Survival Rate by Embarkation Port')
    
    for i, v in enumerate(survival_by_port.values):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # Port distribution
    port_counts = df['Embarked'].value_counts()
    axes[1].pie(port_counts.values, labels=port_counts.index, autopct='%1.1f%%',
               colors=['#ff6b6b', '#4ecdc4', '#95a5a6'])
    axes[1].set_title('Embarkation Port Distribution')
    
    plt.tight_layout()
    return fig

def plot_survival_heatmap(df):
    """Create survival heatmap by class and gender"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    survival_matrix = pd.crosstab(df['Pclass'], df['Sex'], 
                                  values=df['Survived'], aggfunc='mean') * 100
    
    sns.heatmap(survival_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
               cbar_kws={'label': 'Survival Rate (%)'}, ax=ax)
    ax.set_title('Survival Rate Heatmap: Class vs Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Passenger Class')
    ax.set_yticklabels(['First Class', 'Second Class', 'Third Class'])
    
    plt.tight_layout()
    return fig