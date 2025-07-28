import sys, os, pathlib, json
from tqdm import tqdm

try:
    import openslide
except ImportError:
    sys.exit("✖  openslide-python not found – activate environment first")

good, bad = [], []

def probe(path: pathlib.Path):
    try:
        slide = openslide.OpenSlide(str(path))
        # Force-load a small tile to catch deep I/O errors
        _ = slide.read_region((0, 0), 0, (256, 256))
        slide.close()
        return True
    except Exception as e:
        bad.append({"file": str(path), "error": str(e)})
        return False

def main(root_paths):
    svs_files = []
    for root in root_paths:
        root_path = pathlib.Path(root)
        if not root_path.exists():
            print(f"⚠️  Path does not exist: {root}")
            continue
        found = list(root_path.rglob("*.svs"))
        print(f"🔍 Found {len(found)} .svs files in {root}")
        svs_files.extend(found)

    print(f"\n📦 Total .svs files to check: {len(svs_files)}\n")

    for p in tqdm(svs_files, desc="🧪 checking .svs"):
        (good if probe(p) else bad).append(str(p))

    print(f"\n✅ Valid slides: {len(good)}")
    print(f"❌ Corrupted slides: {len(bad)}\n")

    with open("good_svs.txt", "w") as g:
        g.write("\n".join(good))
    with open("bad_svs.txt", "w") as b:
        for item in bad:
            b.write(f"{item['file']}\t{item['error']}\n")

    print("📁 Written to good_svs.txt and bad_svs.txt")



if __name__ == "__main__":
    print("✅ Reached __main__")
    if len(sys.argv) < 2:
        sys.exit("Usage: python check_svs_integrity.py <folder1> [folder2 …]")
    main(sys.argv[1:])


