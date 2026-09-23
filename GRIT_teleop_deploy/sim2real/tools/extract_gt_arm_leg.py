"""Extract only the six explicitly selected non-M GT motions and captions."""
import hashlib
import json
from pathlib import Path
import tarfile

import zstandard

REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO / "HML3D-G1.tar.zst"
DEST = REPO / "GRIT_teleop_deploy/gt_hml3d_arm_leg/source"
SELECTED = {"005225": 1183, "004558": 222, "001987": 1804,
            "012511": 307, "002248": 1587, "005585": 950}


def main():
    wanted = {f"HML3D-G1/{folder}/{mid}.{ext}": DEST / folder / f"{mid}.{ext}"
              for mid in SELECTED for folder, ext in (("texts", "txt"), ("motion", "pkl"))}
    initial = ARCHIVE.stat()
    found = {}
    with ARCHIVE.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                if member.name not in wanted:
                    continue
                if not member.isfile() or member.size > 20_000_000 or member.name in found:
                    raise ValueError(f"Invalid or duplicate selected member: {member.name}")
                data = archive.extractfile(member).read()
                if len(data) != member.size:
                    raise ValueError(f"Incomplete selected member: {member.name}")
                target = wanted[member.name]
                if target.exists():
                    if target.read_bytes() != data:
                        raise ValueError(f"Refusing to overwrite different file: {target}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as out:
                        out.write(data)
                found[member.name] = dict(path=str(target), size=len(data),
                                          sha256=hashlib.sha256(data).hexdigest())
                print(f"Extracted {member.name} ({len(found)}/{len(wanted)})", flush=True)
    final = ARCHIVE.stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise ValueError("Archive changed during extraction; retry after copy completes.")
    if found.keys() != wanted.keys():
        raise ValueError(f"Missing archive members: {wanted.keys() - found.keys()}")
    manifest = dict(archive=str(ARCHIVE), archive_size=final.st_size,
                    archive_mtime_ns=final.st_mtime_ns, selected=SELECTED, members=found,
                    split="unknown; archive contains no split lists",
                    pairing="exact caption matches; non-M versions explicitly selected")
    path = DEST.parent / "extraction.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Existing extraction manifest differs; not overwriting.")
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Ready: {path}")


if __name__ == "__main__":
    main()
