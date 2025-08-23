from typing import Any, Dict, List, Optional
from components.data_structures import RetrievalDoc

def mean_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    n = len(vectors)
    return [sum(vec[i] for vec in vectors) / n for i in range(dim)]

def fuse_vectors(primary: List[float], global_vec: List[float], alpha: float = 0.8) -> List[float]:
    """
    Late-chunking fallback fusion: chunk_vec' = alpha*chunk_vec + (1-alpha)*global_doc_vec
    """
    if not primary:
        return global_vec
    if not global_vec:
        return primary
    d = min(len(primary), len(global_vec))
    return [alpha * primary[i] + (1.0 - alpha) * global_vec[i] for i in range(d)]

def sliding_chunks(tokens: List[str], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(tokens):
        j = min(len(tokens), i + chunk_size)
        spans.append({"start": i, "end": j, "text": " ".join(tokens[i:j])})
        if j == len(tokens):
            break
        i += step
    return spans

def _which_section(start_token: int, section_spans: List[Dict[str, Any]]) -> Optional[int]:
    for i, sec in enumerate(section_spans):
        if sec["start"] <= start_token < sec["end"]:
            return i
    return None

def _pack_results(results: Dict[str, Any]) -> List[RetrievalDoc]:
    ids = results.get("ids", [[]])
    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])
    embs = results.get("embeddings", [[]])
    out: List[RetrievalDoc] = []
    if not ids or not ids[0]:
        return out
    for i in range(len(ids[0])):
        metadata = metas[0][i] if metas and metas[0] and len(metas[0]) > i else {}
        embedding = embs[0][i] if embs and embs[0] and len(embs[0]) > i else None
        out.append(RetrievalDoc(id=ids[0][i], text=docs[0][i], metadata=metadata, embedding=embedding))
    return out
