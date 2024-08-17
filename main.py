from typing import Union
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
from qdrant_client import QdrantClient

from starlette.middleware.cors import CORSMiddleware

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

qdrant_client = QdrantClient(
    url="https://f575a6c0-4880-4ba1-bac4-ce3febd5e83a.us-east4-0.gcp.cloud.qdrant.io",
    api_key="P9qpgcryuCBF8c78hYU4QwImfJIkZimHXZSIBqfax6W3rGi-LNwN4g",
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Replace with your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
class UserCreate(BaseModel):
    query: str


@app.post("/")
def read_root(query_data: UserCreate):
    text = query_data.query

    text_inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    text_emb = model.get_text_features(**text_inputs)


    result = qdrant_client.search(
        collection_name="food_nutrition_data",
        query_vector=text_emb.squeeze().tolist(),
        limit=1,

    )
    return {"nutritions": result}


@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
    try:
        # Read the uploaded file
        contents = await image.read()
        image_file = io.BytesIO(contents)

        # Open the image using PIL
        img = Image.open(image_file)
        img_inputs = processor(images=[img], return_tensors="pt", padding=True).to(device)
        img_emb = model.get_image_features(**img_inputs)

        
        result = qdrant_client.search(
            collection_name="food_nutrition_data",
            query_vector=img_emb.squeeze().tolist(),
            limit=1,

        )
        print(result)
        sharing_data = {"name":result[0].payload["food"],
                        "nutritionalValues":[
                             {"label":"Calories","value":result[0].payload["calories"]},
                             {"label":"Carbohydrates","value":result[0].payload["carbohydrates"]},
                             {"label":"Cholestrol","value":result[0].payload["cholestrol"]},
                             {"label":"Fats","value":result[0].payload["fats"]},
                             {"label":"Protein","value":result[0].payload["proteins"]},
                             ]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return sharing_data
