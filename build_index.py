import torch,open_clip,os
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = "data/images"
OUTPUT_PATH = "data/image_index.pt"
MODEL_PATH = "model.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model,_,preprocess = open_clip.create_model_and_transforms("ViT-B-32",pretrained="openai")
model.load_state_dict(torch.load(MODEL_PATH,map_location=device),strict=False)
model.eval().to(device)

embeddings,paths = [],[]

print(f"Encoding images")
for fname in tqdm(os.listdir(IMAGE_DIR)):
    fpath = os.path.join(IMAGE_DIR,fname)
    try:
        img = Image.open(fpath).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
            feat = feat/feat.norm(dim=-1,keepdim=True)
        embeddings.append(feat.cpu())
        paths.append(fname)
    except Exception as e:
        print(f"Skipping {e}")
if not embeddings:
    raise RuntimeError("No imagess found")
embeddings = torch.cat(embeddings)
torch.save({"embeddings" : embeddings,"paths":paths},OUTPUT_PATH)
print(f"Saved {len(paths)} images to {OUTPUT_PATH}")
print(f"Embedding shape:{embeddings.shape}")