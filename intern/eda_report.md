# Titanic EDA Report - Comprehensive Analysis

## Executive Summary

This report presents findings from Exploratory Data Analysis (EDA) on the Titanic passenger dataset, examining factors that influenced survival during the 1912 disaster.

**Key Findings at a Glance:**
- **Overall Survival Rate:** 38.4%
- **Strongest Predictors:** Gender (74% female vs 19% male survival)
- **Class Impact:** 63% first-class vs 24% third-class survival
- **Age Factor:** Children had 53% survival vs 38% average

## 1. Data Overview

### Dataset Description
- **Records analyzed:** 891 passengers
- **Features examined:** 12 variables
- **Data quality:** Minor missing values (Age: 20%, Embarked: 0.2%)

### Feature Categories
| Category | Features |
|----------|----------|
| Demographic | Age, Sex |
| Socioeconomic | Pclass, Fare |
| Family | SibSp, Parch |
| Administrative | PassengerId, Embarked |

## 2. Univariate Analysis

### Survival Distribution
- **Survived:** 342 passengers (38.4%)
- **Did not survive:** 549 passengers (61.6%)

### Demographics
- **Gender split:** 65% male, 35% female
- **Age range:** 0.42 - 80 years
- **Average age:** 29.7 years

### Socioeconomic Profile
- **Class distribution:** 24% First, 21% Second, 55% Third
- **Average fare:** $32.20
- **Fare range:** $0 - $512

## 3. Bivariate Analysis

### Gender Impact on Survival