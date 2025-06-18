import os
import json
import torch
import clip
import pandas as pd
from PIL import Image

def validate_tags(image_dir, tagged_csv_path, tag_path="tags/tag_list.json", model_name="ViT-B/32", threshold=0.25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    # Load predefined tag dictionary
    with open(tag_path, "r") as f:
        tag_dict = json.load(f)

    # Load tagged image metadata
    df = pd.read_csv(tagged_csv_path)
    validation_results = []

    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, f"{row['id']}.png")
        if not os.path.exists(image_path):
            continue

        # Preprocess and encode image
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        unreliable = []

        for category, tag_list in tag_dict.items():
            # Encode tag texts
            text_inputs = clip.tokenize(tag_list).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze()

            # Get top-k indices by similarity
            top_k = 2 if category == "Relational Expression" else 1
            top_indices = similarity.topk(top_k).indices.cpu().tolist()

            # Combined filtering: not in top-k and below threshold
            for i, tag in enumerate(tag_list):
                score = similarity[i].item()
                if i not in top_indices and score < threshold:
                    unreliable.append(f"{category}: {tag} (score={score:.2f})")

        validation_results.append({
            "id": row["id"],
            "unreliable_tags": "; ".join(unreliable)
        })

    # Save results to CSV
    os.makedirs("outputs", exist_ok=True)
    output_df = pd.DataFrame(validation_results)
    output_df.to_csv("outputs/tagged_results_unreliable.csv", index=False)
    print("Validation complete. Results saved to outputs/tagged_results_unreliable.csv")

if __name__ == "__main__":
    validate_tags(
        image_dir="data/", 
        tagged_csv_path="outputs/tagged_results.csv",  
        tag_path="tags/tag_list.json"
    )

