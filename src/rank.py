import numpy as np, heapq, nltk
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer
from .config import ModelConfig
from .utils import debug

nltk.download('punkt', quiet=True)
_refine_model = None

def focus_keywords(job: str) -> list:
    r = Rake(); r.extract_keywords_from_text(job)
    kws = [p for p in r.get_ranked_phrases() if 1<=len(p.split())<=4][:5]
    debug(f"Keywords: {kws}")
    return kws

def topk(query_vec, docs, k=None):
    k = k or ModelConfig.RETRIEVAL_TOP_K
    candidates = []
    for d in docs.values():
        sims = np.dot(d["embeddings"], query_vec)
        for s,m in zip(sims, d["chunk_metadata"]):
            candidates.append((float(s), m))
    return heapq.nlargest(k, candidates, key=lambda x: x[0])

def rerank(cands, kws, n=None):
    n = n or ModelConfig.FINAL_SECTIONS
    scored = []
    for sim, m in cands:
        boost = 0.0
        heading = m["heading"].lower()
        for kw in kws:
            if kw.lower() in heading:
                boost = 1.0; break
        combined = ModelConfig.SIMILARITY_WEIGHT*sim + ModelConfig.HEADING_BOOST_WEIGHT*boost
        m.update(similarity_score=sim, heading_boost=boost, combined_score=combined)
        scored.append(m)
    top = sorted(scored, key=lambda x: x["combined_score"], reverse=True)[:n]
    debug(f"Selected top {len(top)} sections")
    return top

def refine_text(text: str, qvec: np.ndarray) -> str:
    global _refine_model
    if _refine_model is None:
        _refine_model = SentenceTransformer(
            ModelConfig.EMBEDDING_MODEL,
            trust_remote_code=True  # Add this line
        )
    sents = nltk.sent_tokenize(text)
    if len(sents)<=ModelConfig.TOP_SENTENCES:
        return text
    embs = _refine_model.encode(sents, normalize_embeddings=True)
    sims = np.dot(embs, qvec)
    idx = np.argsort(-sims)[:ModelConfig.TOP_SENTENCES]
    return " ".join(sents[i] for i in idx)

def pipeline(query_vec, docs, job: str):
    kws = focus_keywords(job)
    cands = topk(query_vec, docs)
    top = rerank(cands, kws)
    sections = [{"document":m["document"],"page":m["page"],
                 "section_title":m["heading"],"importance_rank":i+1}
                for i,m in enumerate(top)]
    subs = [{"document":m["document"],"page":m["page"],
             "refined_text":refine_text(m["text"],query_vec)}
            for m in top]
    return sections, subs
