"""Audit exact caption candidates without guessing dataset-index to GT identity."""
from pathlib import Path
import collections
import json
import tarfile
import zstandard

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "GRIT_teleop_deploy"
OUT = BASE / "gt_val_matched500_pairing_audit"


def main():
    samples = json.loads((BASE / "gen_test2576_dar0911_mpjpe_0001_0500/input_manifest.json").read_text())["samples"]
    samples = [s for s in samples if 1 <= s["sample_id"] <= 500]
    assert len(samples) == 500
    prompts = {s["prompt"].strip() for s in samples}
    candidates = collections.defaultdict(list)
    other_files, motion_ids = [], set()
    captions = 0
    with (ROOT / "HML3D-G1.tar.zst").open("rb") as stream:
        with zstandard.ZstdDecompressor().stream_reader(stream) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    p = Path(member.name)
                    if p.parent.name == "motion" and p.suffix == ".pkl":
                        motion_ids.add(p.stem)
                    elif p.parent.name == "texts" and p.suffix == ".txt":
                        text = archive.extractfile(member).read().decode("utf-8-sig")
                        captions += 1
                        for line in text.splitlines():
                            fields = line.split("#")
                            if fields[0].strip() in prompts:
                                candidates[fields[0].strip()].append(dict(gt_id=p.stem,
                                    archive_text=member.name, caption=line,
                                    start_tag=fields[2] if len(fields)>2 else None,
                                    end_tag=fields[3] if len(fields)>3 else None))
                    else:
                        other_files.append(member.name)
    rows = []
    for s in samples:
        matches = [m for m in candidates[s["prompt"].strip()] if m["gt_id"] in motion_ids]
        ids = sorted({m["gt_id"] for m in matches})
        non_m = sorted({i for i in ids if not i.startswith("M")})
        rows.append(dict(sample_id=s["sample_id"], src_idx=s["src_idx"], prompt=s["prompt"],
            src_length=s["src_length"], candidate_ids=ids, non_m_candidate_ids=non_m, captions=matches))
    stats = dict(requested=500, archive_motion_count=len(motion_ids), archive_text_count=captions,
        no_caption_match=sum(not r["candidate_ids"] for r in rows),
        unique_exact_id=sum(len(r["candidate_ids"])==1 for r in rows),
        multiple_exact_ids=sum(len(r["candidate_ids"])>1 for r in rows),
        unique_non_m_id=sum(len(r["non_m_candidate_ids"])==1 for r in rows),
        multiple_non_m_ids=sum(len(r["non_m_candidate_ids"])>1 for r in rows))
    OUT.mkdir(exist_ok=False)
    result = dict(summary=stats, non_motion_text_members=other_files, samples=rows,
        warning="Caption matches are candidates, not proof of original val index or mirror/crop identity. No simulation launched.")
    (OUT / "pairing_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(summary=stats, other_files=other_files[:50], output=str(OUT)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
