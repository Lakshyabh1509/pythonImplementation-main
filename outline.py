import json

def build_outline(lines, preds, confs, label_map, min_conf=0.5):
    outline = []
    for line, pred, conf in zip(lines, preds, confs):
        if label_map[pred] in {"H1", "H2", "H3"} and conf > min_conf:
            outline.append({
                "level": label_map[pred],
                "text": line["text"],
                "page": line["page"] + 1
            })
    return outline

def write_json(output_file, title, outline):
    """
    Write the outline data to a JSON file
    
    Args:
        output_file: Full path to the output JSON file
        title: Document title
        outline: List of heading entries
    """
    out = {"title": title, "outline": outline}
    with open(output_file, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)