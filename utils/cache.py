import hashlib, json, pathlib, shelve, time
from typing import Any, Dict

_CACHE_DIR = pathlib.Path("data/cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_DB = shelve.open(str(_CACHE_DIR / "psg_cache.db"))

def _hash(obj: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def get(key_dict: Dict[str, Any]) -> str | None:
    return _DB.get(_hash(key_dict))

def put(key_dict: Dict[str, Any], value: str):
    _DB[_hash(key_dict)] = value
    _DB.sync()