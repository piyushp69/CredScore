from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pickle
import pandas as pd
import shap
import os

app = FastAPI(title="Credit Risk ML API")

# Load models at startup
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

print("Loading model and explainer...")
with open(os.path.join(models_dir, 'xgboost_model.pkl'), 'rb') as f:
    clf = pickle.load(f)
    
with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)
    
with open(os.path.join(models_dir, 'shap_explainer.pkl'), 'rb') as f:
    explainer = pickle.load(f)

class ApplicantFeatures(BaseModel):
    features: Dict[str, float]

@app.get("/")
def read_root():
    return {"message": "Credit Risk ML API is running"}

@app.post("/predict")
def predict(applicant: ApplicantFeatures):
    try:
        # Create DataFrame aligned with feature_names, missing filled with 0.0
        data_dict = {feat: [applicant.features.get(feat, 0.0)] for feat in feature_names}
        data = pd.DataFrame(data_dict)
        
        # Predict probability of default
        prob_default = clf.predict_proba(data)[0][1]
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(data)
        
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
            
        feature_contributions = {feat: float(val) for feat, val in zip(feature_names, sv)}
        sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
        
        top_3_positive = {k: v for k, v in sorted_features[:3]}
        top_3_negative = {k: v for k, v in sorted_features[-3:]}
        
        return {
            "probability_of_default": float(prob_default),
            "top_3_positive_contributions": top_3_positive,
            "top_3_negative_contributions": top_3_negative
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
