import torch, open_clip
from datasets import load_dataset
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device)

for p in model.visual.parameters():
    p.requires_grad = False
for name, p in model.visual.named_parameters():
    if any(layer in name for layer in ["resblocks.10", "resblocks.11", "proj"]):
        p.requires_grad = True

train_ds = load_dataset("timm/oxford-iiit-pet", split="train")
labels = train_ds.features["label"].names
text_inputs = tokenizer([f"a photo of a {lbl}" for lbl in labels]).to(device)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)  # smooth LR decay
scaler = torch.amp.GradScaler("cuda")

def clip_loss(img_feat, txt_feat):
    logits = img_feat @ txt_feat.T
    targets = torch.arange(len(logits), device=device)
    return (nn.functional.cross_entropy(logits, targets)
          + nn.functional.cross_entropy(logits.T, targets)) / 2

def collate_fn(batch):
    images = [preprocess(x["image"]) for x in batch]
    labels = torch.tensor([x["label"] for x in batch])
    return torch.stack(images), labels

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    model.train()
    epochs, batch_size = 10, 8

    for epoch in range(epochs):
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_fn)
        running_loss = 0.0

        for imgs, lbls in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
            imgs, lbls = imgs.to(device), lbls.to(device)
            txt_batch = text_inputs[lbls]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                img_feat = model.encode_image(imgs)
                txt_feat = model.encode_text(txt_batch)

                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

                loss = clip_loss(img_feat, txt_feat)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), "model.pt")