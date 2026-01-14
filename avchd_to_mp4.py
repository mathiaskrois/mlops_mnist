import subprocess
from pathlib import Path

# Paths
INPUT_DIR = Path("/Users/mathiaskroismoller/Desktop/AVCHD")
OUTPUT_DIR = Path("/Users/mathiaskroismoller/Desktop/out")

# Create output base directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Supported AVCHD extensions
EXTENSIONS = [".mts", ".MTS"]

def convert_file(input_file: Path):
    # Preserve folder structure
    relative_path = input_file.relative_to(INPUT_DIR)
    output_file = OUTPUT_DIR / relative_path.with_suffix(".mp4")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        print(f"Skipping (already exists): {output_file}")
        return

    command = [
        "ffmpeg",
        "-i", str(input_file),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_file)
    ]

    print(f"Converting: {input_file}")
    subprocess.run(command, check=True)

def main():
    for ext in EXTENSIONS:
        for file in INPUT_DIR.rglob(f"*{ext}"):
            convert_file(file)

if __name__ == "__main__":
    main()
