from pathlib import Path

DATASET_DIR = Path("../dataset/raw")

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm"
}


def main():
    if not DATASET_DIR.exists():
        print("Dataset folder not found:")
        print(DATASET_DIR.resolve())
        return

    total_videos = 0
    categories = {}

    for file in DATASET_DIR.rglob("*"):
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS:
            total_videos += 1

            relative = file.relative_to(DATASET_DIR)

            if len(relative.parts) >= 2:
                category = relative.parts[0]
                label = relative.parts[1]
            else:
                category = "Unknown"
                label = file.parent.name

            categories.setdefault(category, {})
            categories[category][label] = (
                categories[category].get(label, 0) + 1
            )

    print("=" * 60)
    print("SIGNBRIDGE AI - DATASET INSPECTION")
    print("=" * 60)

    print(f"\nDataset location:")
    print(DATASET_DIR.resolve())

    print(f"\nTotal videos found: {total_videos}")

    print("\nCategories:")
    print("-" * 60)

    for category, labels in sorted(categories.items()):
        category_total = sum(labels.values())

        print(f"\n{category}: {category_total} videos")

        for label, count in sorted(labels.items()):
            print(f"    {label}: {count}")


if __name__ == "__main__":
    main()