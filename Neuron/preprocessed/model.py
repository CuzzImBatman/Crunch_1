import torch
import torch.nn as nn
import timm
class ImageEncoder(nn.Module):
    def __init__(self,resources_path):
        super().__init__()
        model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)
        
        pretrain_model_path=f'{resources_path}/pytorch_model.bin'
        try:
            model.load_state_dict(torch.load((f'{pretrain_model_path}')), strict=True)
        except:
            model = timm.create_model(
    "hf-hub:bioptimus/H-optimus-0", pretrained=False, init_values=1e-5, dynamic_img_size=True
)           
            model.load_state_dict(torch.load((f'{pretrain_model_path}')), strict=True)
        # model.load_state_dict(torch.load(("D:/Downloads/pytorch_model.bin"), map_location="cuda:0"), strict=True)
        self.model=model
        # self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x=x.unsqueeze(0) # for testing single image
        # print(x.shape)
        x = self.model(x)

        return x
