from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from starlette.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Replace with your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the CLIP model and processor once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
processor = None

@app.on_event("startup")
async def load_model():
    global model, processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Model loaded")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://f575a6c0-4880-4ba1-bac4-ce3febd5e83a.us-east4-0.gcp.cloud.qdrant.io",
    api_key="P9qpgcryuCBF8c78hYU4QwImfJIkZimHXZSIBqfax6W3rGi-LNwN4g",
)

# Define Pydantic model for query data
class UserCreate(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/")
async def create_embeddings(query_data: UserCreate):
    try:
        # Process the input text and generate embeddings
        text = query_data.query
        text_inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
        text_emb = model.get_text_features(**text_inputs)

        # Search in Qdrant
        result = qdrant_client.search(
            collection_name="food_nutrition_data",
            query_vector=text_emb.squeeze().tolist(),
            limit=1,
        )
        return {"nutritions": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
        
    try:
        # Read and process the image file
        contents = await image.read()
        image_file = io.BytesIO(contents)
        img = Image.open(image_file)
        img_inputs = processor(images=[img], return_tensors="pt", padding=True).to(device)
        img_emb = model.get_image_features(**img_inputs)

        # Search in Qdrant
        result = qdrant_client.search(
            collection_name="food_nutrition_data",
            query_vector=img_emb.squeeze().tolist(),
            limit=1,
        )
        
        sharing_data = {
            "name": result[0].payload["food"],
            "nutritionalValues": [
                {"label": "Calories", "value": result[0].payload["calories"]},
                {"label": "Carbohydrates", "value": result[0].payload["carbohydrates"]},
                {"label": "Cholesterol", "value": result[0].payload["cholestrol"]},
                {"label": "Fats", "value": result[0].payload["fats"]},
                {"label": "Protein", "value": result[0].payload["proteins"]},
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return sharing_data
