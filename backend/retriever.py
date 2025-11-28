import torch
from pathlib import Path
import  os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_index(path="data/image_index.pt",device="cuda"):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {path}. Run build_index.py first.")
    data = torch.load(path, map_location=device)
    embeddings = data["embeddings"].to(device)
    paths = data["paths"]
    return embeddings, paths
def search_images(model,tokenizer,prompt,embeddings,paths,device):
    if isinstance(prompt, list):
        text = tokenizer(prompt).to(device)
    else:
        text = tokenizer([prompt]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat = text_feat/text_feat.norm(dim=-1,keepdim=True)
        sims = (text_feat@embeddings.T).squeeze()
    topk = sims.topk(5)
    results = []
    for i in topk.indices:
        fname = os.path.basename(str(paths[i])).replace("\\", "/")
        results.append({
            "path": fname,
            "url": f"/images/{fname}",
            "score": float(sims[i]),
        })
    return results