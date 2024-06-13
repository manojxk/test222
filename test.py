import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

WORLD_CUP_TEAMS = [
    "Afghanistan", "Australia", "Bangladesh", "Canada", "England", "India",
    "Ireland", "Namibia", "Nepal", "Netherlands", "New Zealand", "Oman",
    "Pakistan", "Papua New Guinea", "Scotland", "South Africa", "Sri Lanka",
    "Uganda", "United States of America", "West Indies"
]

def read_file(file_path, file_type='excel'):
    if file_type == 'excel':
        return pd.read_excel(file_path)
    elif file_type == 'csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError("Invalid file type. Supported types are 'excel' and 'csv'.")

def filter_and_select(df, filter_conditions, columns_to_select):
    mask = pd.Series([True] * len(df))
    for col, val in filter_conditions.items():
        if isinstance(val, list):
            mask &= df[col].isin(val)
        else:
            mask &= (df[col] == val)
    
    filtered_df = df[mask]
    selected_columns_df = filtered_df[columns_to_select]
    return selected_columns_df

def merge_dataframes(matches_df, team_stats_df, rank_df):
    merged_df = pd.merge(matches_df, team_stats_df, left_on='Team1', right_on='Team', how='left', suffixes=('', '_Team1'))
    merged_df = merged_df.drop(columns=['Team'])
    merged_df = pd.merge(merged_df, team_stats_df, left_on='Team2', right_on='Team', how='left', suffixes=('_Team1', '_Team2'))
    merged_df = merged_df.drop(columns=['Team'])
    merged_df = pd.merge(merged_df, rank_df, left_on='Team1', right_on='Team', how='left')
    merged_df = merged_df.rename(columns={'Rank': 'Rank_Team1'}).drop(columns=['Team'])
    merged_df = pd.merge(merged_df, rank_df, left_on='Team2', right_on='Team', how='left')
    merged_df = merged_df.rename(columns={'Rank': 'Rank_Team2'}).drop(columns=['Team'])
    return merged_df

def preprocess_data(df):
    df['winner'] = np.where(df['winner'] == df['Team1'], 1, 0)
    df['Rank_Diff'] = df['Rank_Team1'] - df['Rank_Team2']
    df['Win_Ratio_Team1'] = df['Won_Team1'] / (df['Won_Team1'] + df['Lost_Team1'])
    df['Win_Ratio_Team2'] = df['Won_Team2'] / (df['Won_Team2'] + df['Lost_Team2'])
    return df

def build_pipeline():
    categorical_features = ['Team1', 'Team2']
    numeric_features = ['Mat_Team1', 'Win_Ratio_Team1', 'Mat_Team2', 'Win_Ratio_Team2', 'Rank_Team1', 'Rank_Team2', 'Rank_Diff']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )

    return preprocessor

def train_model(X_train, y_train, model, params):
    skf = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(model, params, cv=skf, n_jobs=-1, verbose=True, error_score='raise')
    grid_search.fit(X_train, np.ravel(y_train))
    return grid_search.best_estimator_

def analyze_model(model, X_train, X_test, y_train, y_test):
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_train_proba = model.predict_proba(X_train)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    auc_test = roc_auc_score(y_test, y_test_proba)
    auc_train = roc_auc_score(y_train, y_train_proba)
    
    plt.figure(figsize=(15, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="Test AUC = %.2f" % auc_test)
    plt.plot(fpr_train, tpr_train, label="Train AUC = %.2f" % auc_train)
    plt.legend()
    plt.title('ROC Curve')
    plt.show()
    
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

def main():
    # Read and process data
    file_path_1 = "teams_all_matches.xlsx"
    team_stats_df = read_file(file_path_1)
    filter_conditions_1 = {'Team': WORLD_CUP_TEAMS}
    columns_to_select_1 = ['Team', 'Mat', 'Won', 'Lost']
    team_stats_df = filter_and_select(team_stats_df, filter_conditions_1, columns_to_select_1)
    
    file_path_3 = "teamsdata.csv"
    matches_df = read_file(file_path_3, file_type='csv')
    filter_conditions_3 = {'Team1': WORLD_CUP_TEAMS, 'Team2': WORLD_CUP_TEAMS}
    columns_to_select_3 = ['Team1', 'Team2', 'winner']
    matches_df = filter_and_select(matches_df, filter_conditions_3, columns_to_select_3)

    file_path_4 = "icc_rankings.xlsx"
    rank_df = read_file(file_path_4)
    filter_conditions_4 = {'Team': WORLD_CUP_TEAMS}
    columns_to_select_4 = ['Team', 'Rank']
    rank_df = filter_and_select(rank_df, filter_conditions_4, columns_to_select_4)

    merged_df = merge_dataframes(matches_df, team_stats_df, rank_df)
    df = preprocess_data(merged_df)

    features = ['Team1', 'Team2', 'Mat_Team1', 'Win_Ratio_Team1', 'Mat_Team2', 'Win_Ratio_Team2', 'Rank_Team1', 'Rank_Team2', 'Rank_Diff']
    target = 'winner'

    X = df[features]
    y = df[target]

    preprocessor = build_pipeline()

    gb_params = {
        "gb__learning_rate": [0.01, 0.1, 0.3],
        "gb__min_samples_split": [5, 10, 15],
        "gb__min_samples_leaf": [3, 5, 10],
        "gb__max_depth": [3, 5, 10],
        "gb__max_features": ["sqrt", "log2"],
        "gb__n_estimators": [100, 200, 300]
    }

    rf_params = {
        "rf__max_depth": [10, 20, 30],
        "rf__min_samples_split": [2, 10, 20],
        "rf__max_leaf_nodes": [50, 100, 200],
        "rf__min_samples_leaf": [1, 5, 10],
        "rf__n_estimators": [100, 200, 300],
        "rf__max_features": ["sqrt", "log2"]
    }

    xgb_params = {
        "xgb__learning_rate": [0.01, 0.1, 0.3],
        "xgb__max_depth": [3, 5, 10],
        "xgb__n_estimators": [100, 200, 300],
        "xgb__subsample": [0.7, 0.8, 1.0],
        "xgb__colsample_bytree": [0.7, 0.8, 1.0]
    }

    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('gb', GradientBoostingClassifier(random_state=5))
    ])

    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(random_state=1))
    ])

    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    gb_model = train_model(X_train, y_train, gb_pipeline, gb_params)
    rf_model = train_model(X_train, y_train, rf_pipeline, rf_params)
    xgb_model = train_model(X_train, y_train, xgb_pipeline, xgb_params)

    print("Gradient Boosting Model Analysis:")
    analyze_model(gb_model, X_train, X_test, y_train, y_test)
    print("\nRandom Forest Model Analysis:")
    analyze_model(rf_model, X_train, X_test, y_train, y_test)
    print("\nXGBoost Model Analysis:")
    analyze_model(xgb_model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
