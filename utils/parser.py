import re
from typing import List, Dict

def parse_document(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = content.split("--------------------------------------------------------------------------------")
    parsed_docs = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        data = {}
        title_match = re.search(r"TITLE:\s*(.+)", sec)
        url_match = re.search(r"URL:\s*(.+)", sec)
        categories_match = re.search(r"categories:\s*(.*)", sec)
        tags_match = re.search(r"tags:\s*(.*)", sec)

        data["title"] = title_match.group(1).strip() if title_match else ""
        data["url"] = url_match.group(1).strip() if url_match else ""
        data["categories"] = categories_match.group(1).strip() if categories_match else ""
        data["tags"] = tags_match.group(1).strip() if tags_match else ""

        if "EXCERPT:" in sec:
            data["content_type"] = "excerpt"
            data["content"] = sec.split("EXCERPT:")[1].strip()
        elif "CONTENT:" in sec:
            data["content_type"] = "content"
            data["content"] = sec.split("CONTENT:")[1].strip()
        else:
            data["content_type"] = "unknown"
            data["content"] = ""

        parsed_docs.append(data)

    return parsed_docs
