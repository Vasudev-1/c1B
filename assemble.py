import json
from pathlib import Path
from datetime import datetime
from .utils import error

def build_output(path: Path, docs: list, persona: str, job: str, sections: list, subs: list):
    try:
        out = {
            "metadata": {
                "documents": docs,
                "persona": persona,
                "job_to_be_done": job,
                "processed_at": datetime.utcnow().isoformat()+"Z"
            },
            "extracted_sections": sections,
            "subsection_analysis": subs
        }
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    except Exception as e:
        error(f"Failed writing JSON to {path}", e)
