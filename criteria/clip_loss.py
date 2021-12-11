import torch
import clip
from models.MultilingualCLIP import multilingual_clip


class CLIPLoss(torch.nn.Module):
    """CLIP Loss Definition"""

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.text_model = multilingual_clip.load_model("M-BERT-Base-ViT-B")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=32)

    def encode_image(self, img):
        img = self.avg_pool(self.upsample(img))
        image_emb = self.model.encode_image(img).float()

        return image_emb

    def encode_text(self, txt):
        return self.text_model(args.description).cuda()

    def forward(self, img, txt):

        img_emb = self.encode_image(img)
        txt_emb = self.encode_text(txt)

        # normalized features
        image_features = img_emb / img_emb.norm(dim=-1, keepdim=True)
        text_features = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        self.logit_scale = self.model.logit_scale.exp()
        logits_per_image = self.logit_scale * image_features @ text_features.t()
        logits_per_text = self.logit_scale * text_features @ image_features.t()

        similarity = 1 - logits_per_image / 100
        return similarity
