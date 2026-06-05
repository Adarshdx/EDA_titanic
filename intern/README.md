# Titanic EDA Project - Exploratory Data Analysis

## 📊 Project Overview
This project performs comprehensive Exploratory Data Analysis (EDA) on the Titanic dataset to uncover patterns, trends, and factors that influenced passenger survival.

## 🎯 Key Objectives
- Analyze demographic patterns (age, gender, class)
- Identify key survival factors
- Visualize correlations and trends
- Generate actionable insights

## 📁 Dataset Information
- **Source**: Titanic passenger list
- **Features**: 12 variables including age, sex, passenger class, fare, survival status
- **Records**: 891 passengers

## 🔍 Analysis Performed

### Statistical Summaries
- Descriptive statistics for numerical features
- Frequency distributions for categorical variables
- Missing data analysis

### Visualizations
1. **Survival Distribution** - Overall survival rates
2. **Class Analysis** - Survival by passenger class
3. **Gender Impact** - Male vs female survival rates
4. **Age Patterns** - Age distribution and survival
5. **Fare Analysis** - Ticket price vs survival
6. **Family Impact** - Family size effect on survival
7. **Embarkation Port** - Port-wise survival rates
8. **Correlation Heatmap** - Feature relationships

### Key Findings
1. **Gender**: 74% female survival vs 19% male survival
2. **Class**: 63% first-class survival vs 24% third-class
3. **Age**: Children (under 10) had higher survival rates
4. **Fare**: Higher fares correlated with better survival
5. **Family**: Small families (1-3 members) had better survival

## 🛠️ Technologies Used
- Python 3.9+
- Pandas - Data manipulation
- NumPy - Numerical operations
- Matplotlib/Seaborn - Visualization
- Jupyter Notebook - Interactive analysis

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/Adarshdx/titanic-eda-project.git
cd titanic-eda-project

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook notebooks/01_eda_analysis.ipynb