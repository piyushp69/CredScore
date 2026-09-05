import os
import gc
import pickle
import pandas as pd
import xgboost as xgb
import shap

def main():
    print("Loading processed dataset...")
    dataset_dir = 'dataset'
    models_dir = 'models'
    
    df = pd.read_parquet(os.path.join(dataset_dir, 'processed_train.parquet'))
    
    y = df['TARGET']
    X = df.drop(columns=['TARGET'])
    
    feature_names = X.columns.tolist()
    
    print("Training XGBoost Classifier on GPU...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist',
        device='cuda',
        random_state=42
    )
    
    clf.fit(X, y)
    
    print("Saving model and feature names...")
    with open(os.path.join(models_dir, 'xgboost_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
        
    with open(os.path.join(models_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
        
    print("Generating SHAP Explainer...")
    explainer = shap.TreeExplainer(clf)
    
    print("Saving SHAP Explainer...")
    with open(os.path.join(models_dir, 'shap_explainer.pkl'), 'wb') as f:
        pickle.dump(explainer, f)
        
    print("Phase 2 complete. Models and explainer saved to models/ directory.")

if __name__ == "__main__":
    main()
