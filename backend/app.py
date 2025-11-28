from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware
from backend.model_loader import load_model
from backend.retriever import load_index, search_images

app = FastAPI()
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images
app.mount("/images", StaticFiles(directory="data/images"), name="images")
model, tokenizer, preprocess, device = load_model()
embeddings, paths = load_index()
@app.get("/search")
def search(prompt: str=Query(...,description="Prompt for retrieval")):
    results = search_images(model, tokenizer,prompt,embeddings,paths,device)
    return {"query":prompt,"results":results}
@app.get("/")
def root():
    return {"message": "Welcome to Pet Image Retriever API"}
