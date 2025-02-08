import numpy as np
import torch
import timm
from huggingface_hub import login
from torchvision import transforms

def get_uni_feature(slide,patch_size,coords):
    
    model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    num_images = len(coords)
    num_features = 1024
    feature_record = np.zeros((num_images, num_features), dtype=np.float32)
    with torch.no_grad():
        for idx, coord in enumerate(coords):
            print(str(idx)+'/'+str(len(coords)))
            print(patch_size)
            patch = slide.read_region(tuple(coord),0,(patch_size,patch_size)).convert('RGB')
            image = transform(patch).unsqueeze(dim=0)
            feature_record[idx,:] = model(image).detach().cpu().numpy().flatten()
            del patch, image 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 
    return feature_record



def get_patch_uni_feature(patches):
    
    if 'model' not in globals():
        global model
        model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    model = model.to('cuda')
    model.eval()
    features = []
    with torch.no_grad():
        # image = transform(patch).unsqueeze(dim=0)
        for patch in patches:
            patch = patch.to('cuda').unsqueeze(dim=0)
            feature = model(patch).detach().cpu().numpy().flatten()
            features.append(feature)
    return features

