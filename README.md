🐶🐱 CLIP-Based Pet Image Retriever

Fine-tuning of OpenAI’s CLIP (ViT-B/32) model on the Oxford-IIIT Pet Dataset to build a text-to-image retrieval system — where typing “bulldog” returns bulldog images and “Persian cat” returns Persian cats.

🚀 Overview

This project fine-tunes CLIP for breed-level image retrieval of cats and dogs.
The pretrained CLIP model achieved 71.4% zero-shot accuracy,
which improved to 85.2% after fine-tuning.

🧠 Fine-Tuning Details

Base model: CLIP ViT-B/32 (openai weights)

Layers fine-tuned:

Vision encoder blocks 10 & 11

Projection layer

Text encoder: kept frozen

Epochs: 10

Dataset: Oxford-IIIT Pet Dataset
 via Hugging Face datasets

Hardware: 4GB GPU compatible
<pre><code>
🧩 Repository Structure
CLIP_based_image_retriever/
│
├── backend/
│   ├── app.py               # FastAPI backend (serves /search endpoint)
│   ├── retriever.py         # CLIP similarity search logic
│   ├── model_loader.py      # Loads model + preprocess
│
├── frontend/
│   ├── index.html           # Web UI for search
│   ├── script.js            # Handles prompt → backend call
│
├── test_data.py             # Downloads and saves Oxford-IIIT Pet test images
├── build_index.py           # Builds CLIP vector store (image embeddings)
├── finetune.py              # Fine-tunes CLIP model
├── data/
│   ├── images/              # Holds images downloaded by test_data.py
│   └── image_index.pt       # Saved image embeddings
│
└── README.md
</code></pre>

⚙️ Setup
1️⃣ Clone the repository
git clone https://github.com/<your-username>/CLIP_based_image_retriever.git
cd CLIP_based_image_retriever

2️⃣ Install dependencies
pip install -r requirements.txt

🐾 Dataset Preparation

Download the Oxford-IIIT Pet test split and save locally:

python test_data.py


This creates a folder:
<pre><code>
data/images/
├── 0_1.jpg
├── 1_3.jpg
├── 2_7.jpg
...
</code></pre>
🧮 Build the Image Index

Next, encode the images using CLIP to create a searchable vector store:

python build_index.py


This generates:

data/image_index.pt

🧠 Fine-Tune CLIP

Fine-tune the image encoder on pet breeds to improve retrieval accuracy:

python finetune.py


After training for 10 epochs, the accuracy improves from 71.4% → 85.2%.

🌐 Run the Backend

Start the FastAPI server:

uvicorn backend.app:app --reload


This serves:

/search?prompt=bulldog → returns top 5 most similar images

/images/<filename>.jpg → serves images statically

💻 Launch the Frontend

Open your web app (for example via a local server):

python -m http.server 5500


Then go to:

http://127.0.0.1:5500/frontend/index.html


Enter a prompt like:

"German shepherd"

and you’ll see top matching dog images instantly retrieved from your fine-tuned model.

📊 Results
Model	Accuracy	Notes
CLIP (zero-shot)	71.4%	Pretrained ViT-B/32
Fine-tuned CLIP	85.2%	Vision encoder partially fine-tuned
🧰 Tech Stack

PyTorch — Deep learning & fine-tuning

OpenCLIP — CLIP model implementation

FastAPI — REST backend

HTML + JS — Simple frontend

Hugging Face Datasets — Data loading

Uvicorn — App server

📁 Required Structure Before Running
<pre><code>
data/
├── images/          # from test_data.py
└── image_index.pt   # from build_index.py 
</code></pre>

