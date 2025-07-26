import argparse, json
from pathlib import Path
from .config import setup_logging
from .utils import validate_input
from .ingest import ingest_documents
from .embed import embed_docs, embed_query
from .rank import pipeline, focus_keywords
from .assemble import build_output

def run_collection(col_path: Path):
    cfg = json.loads((col_path/"challenge1b_input.json").read_text())
    if not validate_input(cfg): return
    pdfs = [col_path/"PDFs"/d["filename"] for d in cfg["documents"]]
    persona, job = cfg["persona"], cfg["job_to_be_done"]

    docs = ingest_documents([str(p) for p in pdfs])
    docs = embed_docs(docs)
    qvec = embed_query(persona, job, focus_keywords(job))
    secs, subs = pipeline(qvec, docs, job)

    build_output(col_path/"challenge1b_output.json",
                 [p.name for p in pdfs], persona, job, secs, subs)
    print(f"{col_path.name} ✓")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base",  default="Challenge_1b")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    setup_logging(debug=args.debug)
    base = Path(args.base)
    for col in sorted(base.iterdir()):
        if col.is_dir() and col.name.startswith("Collection"):
            run_collection(col)

if __name__=="__main__":
    main()
