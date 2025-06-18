import torch
import clip
from PIL import Image
import json

class TagGenerator:
    def __init__(self, tag_path="tags/tag_list.json", device=None, model_name="ViT-B/32"):
        """
        Load CLIP model and tag definitions.

        Parameters:
        - tag_path: Path to JSON file containing categorized tags
        - device: 'cuda' or 'cpu'
        - model_name: Name of CLIP model to use
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        with open(tag_path, "r") as f:
            self.tag_dict = json.load(f)  

    def generate_tags(self, image_path, description_text):
        """
        Generate semantic tags for an image based on visual content and description.

        Parameters:
        - image_path: Path to the image file
        - description_text: Text description associated with the image

        Returns:
        - tags: Dict of selected tags by category
        """
        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        selected_tags = {}

        for category, tag_list in self.tag_dict.items():
            text_inputs = clip.tokenize(tag_list).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text_inputs)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).squeeze()

            top_k = 2 if category == "Relational Expression" else 1
            top_indices = similarity.topk(top_k).indices.cpu().tolist()

            selected_tags[category] = [tag_list[i] for i in top_indices]

        return selected_tags
