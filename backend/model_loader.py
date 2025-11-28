import torch,open_clip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_model():
    model,_,preprocess = open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai')
    model.load_state_dict(torch.load("model.pt",map_location=device),strict=False)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model,tokenizer,preprocess,device
