from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent.parent / "dataset"

def extract_path(topic, advanced, extended):
    if topic == "documents":
        path = DATA_DIR / "documents"
    elif topic == "inquiries":
        path = DATA_DIR / "inquiries"
    else:
        raise ValueError(f"유효하지 않는 주제: {topic}")

    if advanced:
        next_path_str = "advanced"
    else:
        next_path_str = "basic"

    if extended:
        next_path_str += "_merge"

    path = path / next_path_str

    return path