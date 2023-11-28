"""This script traverses a directory and generates a jpg for every file
matched."""
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def main(d: Path = Path(__file__).parents[1] / "rsc",
         glob: str = "**/result.tif"):
    for fp in tqdm(d.glob(glob)):
        if (fp_jpg := fp.with_suffix(".jpg")).exists():
            print(f"Skipping {fp_jpg}")
            continue

        print(f"Generating {fp_jpg}")
        Image.open(fp).convert("RGB").save(fp_jpg)


if __name__ == '__main__':
    main()
