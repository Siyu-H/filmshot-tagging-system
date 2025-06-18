import os
import pandas as pd
from utils.pdf_parser import extract_shot_descriptions_from_pdf
from utils.tag_generator import TagGenerator

# Parse PDF and extract shot description info
pdf_path = "data/Master Shots Volume 2 Shooting Great Dialogue Scenes (Christopher Kenworthy).pdf"
df = extract_shot_descriptions_from_pdf(pdf_path)

# Save structured shot descriptions for Task 1
df.to_csv("outputs/shot_descriptions.csv", index=False)

# Initialize the tag generator
tagger = TagGenerator(
    model_name="ViT-B/32",
    tag_path="tags/tag_list.json"
)

# Generate tags for each image variant
results = []
image_dir = "data"
for i, row in df.iterrows():
    variants = ['a', 'b', 'c'] 
    for suffix in variants:
        image_filename = f"{row['id']}{suffix}.png"
        image_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(image_path):
            continue

        tags = tagger.generate_tags(image_path, row["description"])
        results.append({
            "id": f"{row['id']}{suffix}",
            "shot_title": row["shot_title"],
            "description": row["description"],
            "tags": ", ".join([f"{k}: {', '.join(v)}" for k, v in tags.items()])
        })

# Save results
os.makedirs("outputs", exist_ok=True)
output_df = pd.DataFrame(results)
output_df.to_csv("outputs/tagged_results.csv", index=False)

print("Tagging completed. Output saved to outputs/tagged_results.csv")
