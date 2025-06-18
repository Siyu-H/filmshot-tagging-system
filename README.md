# Automated Image Tagging and Semantic Retrieval from Film Materials

## Introduction

This project simulates an AI-assisted visual storytelling pipeline by converting professional filmmaking materials into a structured, searchable image dataset. The complete system spans from parsing descriptive PDFs to tagging and validating cinematic images with semantic labels, culminating in a searchable format that supports natural language queries.

---

## Task 1: PDF Parsing & Key Frame Extraction

### Objective

Automatically extract structured shot descriptions from a filmmaking reference PDF and retrieve corresponding key frames.

### Implementation

* The `pdf_parser.py` script utilizes `pdfplumber` to read selected pages from the book *Master Shots Volume 2*.
* A regular expression identifies entries formatted as `"1.1 WALKING AND TALKING"` followed by descriptive text.
* Extracted data includes:

  * `id` (e.g., 1.1)
  * `shot_title` (e.g., WALKING AND TALKING)
  * `description` (narrative explanation)
* The parsed information is saved as a pandas DataFrame and used to simulate downstream key frame matching with image files named `{id}.png`.

### Output

A structured CSV with columns: `id`, `shot_title`, and `description`.

---

## Task 2: Semantic Image Tagging

### Objective

Automatically assign semantic tags to each image using both visual and textual input.

### Implementation

* The `tag_generator.py` script uses OpenAI's CLIP model to embed both image features and candidate tags.
* Tags are organized into 7 categories (6 required + 1 relational) and stored in `tags/tag_list.json`.
* For each input pair:

  * The image and description are encoded separately.
  * The model computes cosine similarity between each encoded input and all candidate tags.
  * One top tag is selected from each required category, and two from the Relational Expression category.
* Output is saved to `outputs/tagged_results.csv`.

### Output

A CSV file mapping each image to its corresponding tags while following the defined rules strictly.

---

## Task 3: Output Design for Text-Based Image Search

### Objective

Allow users to retrieve images based on a natural language query.

### Implementation

* The `search.py` module enables lightweight keyword-based search.
* Workflow:

  1. A user inputs a phrase (e.g., "two people arguing").
  2. Keywords are extracted and matched against the tags of each image.
  3. Images are ranked by tag overlap and presented to the user.

### Output

Terminal output showing the top K matched images with scores and relevant tags.

---

## Task 4: Lightweight Tag Calibration

### Objective

Identify potentially unreliable tags through automated validation.

### Implementation

* The `tag_validator.py` script reuses CLIP to check how well each assigned tag matches the image content alone.
* For each image:

  * Similarity scores are computed between the image and each category's tags.
  * Tags not in the top-K or scoring below a threshold (e.g., 0.25) are flagged.
* Results are saved to `outputs/tagged_results_unreliable.csv`.

### Output

A CSV listing all image IDs and their corresponding low-confidence tags.

---

## System Execution Overview

### Modules

* `pdf_parser.py`: Parses PDF and extracts shot descriptions.
* `tag_generator.py`: Applies semantic tagging using image and description.
* `tag_validator.py`: Flags low-confidence tags.
* `search.py`: Implements a text-to-image search logic.
* `main.py`: Orchestrates the full pipeline.

### Workflow

1. Parse and extract descriptions.
2. Match and tag images using CLIP.
3. Validate tag reliability (optional).
4. Search via tag overlap with query terms.

### Folder Structure

```
filmshot-tagging-system/
├── data/                     # Input images
├── outputs/                 # Tagged and validation CSVs
├── tags/                   # JSON file with tag list
├── pdf_parser.py
├── tag_generator.py
├── tag_validator.py
├── search.py
├── main.py
```

---

## Conclusion

This project demonstrates a modular, automated pipeline that turns descriptive film text into searchable visual data. By leveraging CLIP for vision-language understanding, it ensures semantic consistency in tagging and enables intuitive user interactions through natural language queries. A lightweight validation mechanism further ensures tag quality without requiring manual annotation. Overall, the system is practical, extendable, and grounded in real-world AI storytelling needs.
