import torch
import open_clip
import torchvision.transforms as T
from typing import Dict, List


class OpenCLIPPreferenceProxy:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", device: torch.device | None = None):
        if device is None:
            device = torch.device('cuda')
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def _encode_texts(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def build_user_prompt_bank(self, user_to_prompts: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        bank: Dict[str, torch.Tensor] = {}
        for user, prompts in user_to_prompts.items():
            bank[user] = self._encode_texts(prompts)
        return bank

    def score_images(self, images: torch.Tensor, user_prompt_bank: Dict[str, torch.Tensor], temperature: float = 10.0) -> Dict[str, torch.Tensor]:
        to_pil = T.ToPILImage()
        processed = torch.stack([self.preprocess(to_pil(img.cpu())) for img in images])
        processed = processed.to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores: Dict[str, torch.Tensor] = {}
        for user, text_features in user_prompt_bank.items():
            sims = image_features @ text_features.t()
            probs = torch.softmax(temperature * sims, dim=1)
            score = probs.max(dim=1).values
            scores[user] = score
        return scores


