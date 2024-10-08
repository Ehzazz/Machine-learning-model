import joblib  # Make sure to have joblib installed or use pickle if that’s what you prefer
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Load your trained model (change 'your_model.pkl' to your actual model filename)
model = joblib.load("model.pkl")

# Define the input data model
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float


# Create the FastAPI app
app = FastAPI()

# Define a route for prediction
@app.post("/predict")
def predict(data: InputData):
    # Create a feature array from the input data
    features = [[
        data.feature1,
        data.feature2,
        data.feature3,
        data.feature4,
        data.feature5
        
    ]]
    
    # Make prediction using the model
    prediction = model.predict(features)
    
    # Return the prediction as a JSON response
    return JSONResponse(content={"prediction": prediction[0]})  # Assuming prediction is a single value

# Optional: Define a root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Prediction API!"}
