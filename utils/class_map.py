"""Shared class map loading and saving utilities for live-audio-classifier."""
import json
from pathlib import Path
from typing import List
from utils.logging import get_logger

logger = get_logger("class_map")


def load_class_map(data_root: Path, artifacts_dir: Path) -> List[str]:
    """
    Load class names in index order.
    
    Priority:
    1. artifacts/class_map.json with key 'idx2name' (list or dict)
    2. UrbanSound8K.csv to derive classID->name and sort by classID
    
    Args:
        data_root: Path to UrbanSound8K root directory.
        artifacts_dir: Path to artifacts directory (where class_map.json would be).
    
    Returns:
        List of class names in index order (idx 0, 1, 2, ...).
    
    Raises:
        FileNotFoundError: If neither class_map.json nor UrbanSound8K.csv is found.
    
    Example:
        >>> from pathlib import Path
        >>> from utils.class_map import load_class_map
        >>> idx2name = load_class_map(Path("/path/to/UrbanSound8K"), Path("artifacts"))
        >>> print(idx2name[0])  # First class name
    """
    # Try to load from class_map.json first
    cm_path = artifacts_dir / "class_map.json"
    if cm_path.exists():
        try:
            with open(cm_path, "r") as f:
                data = json.load(f)
            if isinstance(data.get("idx2name"), list):
                logger.debug(f"Loaded class map from {cm_path} (list format)")
                return data["idx2name"]
            elif isinstance(data.get("idx2name"), dict):
                # Convert dict to list, sorted by key
                idx2name = [v for k, v in sorted(data["idx2name"].items(), key=lambda kv: int(kv[0]))]
                logger.debug(f"Loaded class map from {cm_path} (dict format)")
                return idx2name
        except Exception as e:
            logger.warning(f"Failed to load class_map.json: {e}, falling back to CSV")
    
    # Fallback: read from UrbanSound8K metadata CSV
    import pandas as pd
    meta1 = data_root / "UrbanSound8K.csv"
    meta2 = data_root / "metadata" / "UrbanSound8K.csv"
    
    if meta1.exists():
        meta_path = meta1
    elif meta2.exists():
        meta_path = meta2
    else:
        raise FileNotFoundError(
            "Could not find UrbanSound8K.csv under data_root or data_root/metadata. "
            f"Also checked for class_map.json at {cm_path}"
        )
    
    df = pd.read_csv(meta_path)
    id_to_name = df.drop_duplicates(subset=["classID"])[["classID", "class"]].sort_values("classID")
    idx2name = id_to_name["class"].tolist()
    logger.debug(f"Loaded class map from {meta_path}")
    return idx2name


def save_class_map(artifacts_dir: Path, idx2name: List[str]) -> None:
    """
    Save class map to artifacts/class_map.json.
    
    Args:
        artifacts_dir: Path to artifacts directory.
        idx2name: List of class names in index order.
    
    Example:
        >>> from pathlib import Path
        >>> from utils.class_map import save_class_map
        >>> save_class_map(Path("artifacts"), ["air_conditioner", "car_horn", ...])
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cm_path = artifacts_dir / "class_map.json"
    
    data = {
        "idx2name": idx2name,
        "num_classes": len(idx2name)
    }
    
    with open(cm_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved class map to {cm_path} ({len(idx2name)} classes)")

