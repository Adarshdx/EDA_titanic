import pandas as pd
import numpy as np

def load_titanic_data():
    """Load and prepare Titanic dataset"""
    # Create sample dataset (in real scenario, you'd load from CSV)
    data = {
        'PassengerId': range(1, 892),
        'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1] * 89 + [0, 1],
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2] * 89 + [1, 3],
        'Name': ['Passenger_' + str(i) for i in range(1, 892)],
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 
                'male', 'male', 'female', 'female'] * 89 + ['male', 'female'],
        'Age': np.random.normal(30, 14, 891).astype(int),
        'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], 891, p=[0.7, 0.2, 0.05, 0.03, 0.01, 0.01]),
        'Parch': np.random.choice([0, 1, 2, 3, 4, 5], 891, p=[0.75, 0.15, 0.05, 0.03, 0.01, 0.01]),
        'Fare': np.random.exponential(30, 891),
        'Embarked': np.random.choice(['S', 'C', 'Q'], 891, p=[0.7, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic patterns
    # First class passengers pay higher fares
    df.loc[df['Pclass'] == 1, 'Fare'] = np.random.uniform(50, 200, len(df[df['Pclass'] == 1]))
    df.loc[df['Pclass'] == 2, 'Fare'] = np.random.uniform(20, 50, len(df[df['Pclass'] == 2]))
    df.loc[df['Pclass'] == 3, 'Fare'] = np.random.uniform(5, 20, len(df[df['Pclass'] == 3]))
    
    # Survival patterns
    df['Survived'] = 0
    # Women survive more
    df.loc[(df['Sex'] == 'female'), 'Survived'] = np.random.choice([0, 1], len(df[df['Sex'] == 'female']), p=[0.26, 0.74])
    # Men survive less
    df.loc[(df['Sex'] == 'male'), 'Survived'] = np.random.choice([0, 1], len(df[df['Sex'] == 'male']), p=[0.81, 0.19])
    # First class better survival
    df.loc[(df['Pclass'] == 1) & (df['Sex'] == 'male'), 'Survived'] = np.random.choice([0, 1], len(df[(df['Pclass'] == 1) & (df['Sex'] == 'male')]), p=[0.63, 0.37])
    df.loc[(df['Pclass'] == 1) & (df['Sex'] == 'female'), 'Survived'] = np.random.choice([0, 1], len(df[(df['Pclass'] == 1) & (df['Sex'] == 'female')]), p=[0.03, 0.97])
    
    return df

def get_data_info(df):
    """Get basic information about the dataset"""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    return info

def clean_data(df):
    """Clean and prepare data for analysis"""
    df_clean = df.copy()
    
    # Fill missing ages with median
    if df_clean['Age'].isnull().any():
        df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    
    # Fill missing embarked with mode
    if df_clean['Embarked'].isnull().any():
        df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    
    # Create family size feature
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    
    # Create age groups
    df_clean['AgeGroup'] = pd.cut(df_clean['Age'], 
                                   bins=[0, 12, 18, 35, 60, 100],
                                   labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])
    
    return df_clean