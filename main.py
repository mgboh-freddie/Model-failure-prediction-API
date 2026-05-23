from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Motor Failure Prediction API",
    description="Predicts electric motor failures using XGBoost",
    version="1.0"
)

# ========== LOAD MODEL ==========
try:
    model = joblib.load('failure_calibrated.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

best_threshold = 0.70  # Set based on validation results (adjust as needed)
# ========== DATA MODEL ==========
class MotorData(BaseModel):
    air_temp: float = Field(..., gt=290, lt=310)
    process_temp: float = Field(..., gt=300, lt=315)
    rotational_speed: float = Field(..., gt=1000, lt=3000)
    torque: float = Field(..., gt=10, lt=80)
    tool_wear: float = Field(..., gt=0, lt=260)

# ========== ROUTES ==========
@app.get("/")
def root():
    return {"message": "Motor Failure Prediction API", "status": "running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_failure(data: MotorData):
    # Check if model loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create engineered features (MUST match training!)
        torque_rotationspeed = data.torque * data.rotational_speed
        torque_toolwear = data.torque * data.tool_wear
        
        # Create feature array in EXACT order model expects
        features = np.array([[
            data.air_temp,
            data.process_temp,
            data.rotational_speed,
            data.torque,
            data.tool_wear,
            torque_rotationspeed,
            torque_toolwear
        ]])
        
        # Predict
        probability = model.predict_proba(features)[0][1]
        prediction = 1 if probability > best_threshold else 0
        
        return {
            "failure_predicted": bool(prediction),
            "failure_probability": round(float(probability), 3),
            "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW",
            "features_used": {
                "air_temp": data.air_temp,
                "process_temp": data.process_temp,
                "rotational_speed": data.rotational_speed,
                "torque": data.torque,
                "tool_wear": data.tool_wear,
                "torque_rotationspeed": torque_rotationspeed,
                "torque_toolwear": torque_toolwear
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

