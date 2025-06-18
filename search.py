import pandas as pd
import json

def load_all_tags(tag_path="tags/tag_list.json"):
    """Flatten all tags into a list"""
    with open(tag_path, "r") as f:
        tag_dict = json.load(f)
    flat_tags = [tag.lower() for tags in tag_dict.values() for tag in tags]
    return flat_tags

def match_query_to_tags(query, tag_candidates):
    """Return matched tags from query"""
    matched = [tag for tag in tag_candidates if tag in query.lower()]
    return matched

def score_tag_matches(row_tags, matched_tags):
    """Count how many matched tags appear in row's tags"""
    score = 0
    for tag in matched_tags:
        if tag in row_tags.lower():
            score += 1
    return score

def search_images(query, tag_csv="outputs/tagged_results.csv", tag_json="tags/tag_list.json", top_k=5):
    df = pd.read_csv(tag_csv)
    tag_candidates = load_all_tags(tag_json)
    matched_tags = match_query_to_tags(query, tag_candidates)

    if not matched_tags:
        print("No tags matched from query.")
        return []

    df["score"] = df["tags"].apply(lambda t: score_tag_matches(t, matched_tags))
    top_results = df[df["score"] > 0].sort_values("score", ascending=False).head(top_k)

    print(f"\nQuery matched tags: {matched_tags}")
    print(f"\nTop {len(top_results)} results:\n")
    print(top_results[["id", "shot_title", "score", "tags"]])

    return top_results

if __name__ == "__main__":
    user_query = input("Enter a description to search (e.g., 'two people arguing'): ")
    search_images(user_query)
