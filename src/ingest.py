import fitz, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict

from .config import ModelConfig
from .utils import debug, error

def extract_pages(pdf_path: str) -> List[str]:
    try:
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text", sort=True) for page in doc]
        debug(f"Extracted {len(pages)} pages from {pdf_path}")
        return pages
    except Exception as e:
        error(f"Error reading PDF {pdf_path}", e)
        return []

def detect_headings(pages: List[str]) -> List[Dict]:
    headings = []
    for i, text in enumerate(pages):
        for ln in filter(None, text.splitlines()):
            lvl = None
            if re.match(r"^\d+(\.\d+){0,2}\s+", ln):
                lvl = f"H{ln.count('.')+1}"
            elif ln.isupper() and 3<=len(ln.split())<=10:
                lvl = "H1"
            elif ln.endswith(":") and 2<=len(ln.split())<=8:
                lvl = "H2"
            elif len(ln.split())<=6 and ln[0].isupper() and not ln.endswith("."):
                lvl = "H2"
            if lvl:
                headings.append({"level":lvl,"text":ln,"page":i+1})
    debug(f"Detected {len(headings)} headings")
    return headings

def chunk_text(full: str) -> List[str]:
    words = full.split()
    w, s = ModelConfig.CHUNK_SIZE, ModelConfig.CHUNK_STRIDE
    chunks = [" ".join(words[i:i+w]) for i in range(0,len(words),s) if len(words[i:i+w])>10]
    debug(f"Created {len(chunks)} chunks")
    return chunks

def process_document(path: str) -> Dict:
    pages = extract_pages(path)
    headings = detect_headings(pages)
    full_text = "\n\n".join(pages)
    chunks = chunk_text(full_text)
    meta = []
    for i, c in enumerate(chunks):
        pos = full_text.find(c)
        page_idx = min(len(pages), pos//max(len(full_text)//len(pages),1)+1)
        heading = next((h["text"] for h in headings if h["page"]<=page_idx), "Unknown")
        meta.append({"chunk_id":i,"page":page_idx,"heading":heading,
                     "text":c,"document":Path(path).name})
    return {"document":Path(path).name,"chunks":chunks,"chunk_metadata":meta}

def ingest_documents(paths: List[str]) -> Dict[str,Dict]:
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = ex.map(process_document, paths)
        return {r["document"]:r for r in results}
