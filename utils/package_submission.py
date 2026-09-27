"""
Packaging script for ML Challenge 2026 final submission zip.

Builds <team_name>_submission.zip matching the required competition structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import sys
import os
import zipfile
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def package_submission(team_name: str):
    zip_filename = f"{team_name}_submission.zip"
    zip_path = PROJECT_ROOT / zip_filename

    print(f"Creating submission package: {zip_filename}...")

    files_to_pack = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
        ("Documentation_template.md", "Documentation_template.md"),
    ]

    # Add all src files
    src_dir = PROJECT_ROOT / "code" / "business_entity_resolution" / "src"
    for src_file in src_dir.glob("*.py"):
        rel_path = f"code/business_entity_resolution/src/{src_file.name}"
        files_to_pack.append((rel_path, rel_path))

    # Verify required files exist
    missing = []
    for rel_path, _ in files_to_pack:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            missing.append(rel_path)

    if missing:
        print(f"❌ Error: Missing required files for submission zip:\n  " + "\n  ".join(missing))
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path, arc_name in files_to_pack:
            full_path = PROJECT_ROOT / rel_path
            print(f"  Adding: {arc_name} ({full_path.stat().st_size / 1024**2:.2f} MB)")
            zf.write(full_path, arcname=arc_name)

    print(f"\n✅ Successfully created: {zip_path} ({zip_path.stat().st_size / 1024**2:.1f} MB)")
    print("Ready for upload to the competition portal!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package ML Challenge 2026 Submission Zip")
    parser.add_argument("--team-name", default="TopTier_Innovators", help="Your team name for the zip filename")
    args = parser.parse_args()
    package_submission(args.team_name)
