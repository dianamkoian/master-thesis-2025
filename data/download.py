"""Скачивание полного датасета Ozon eCup 2025 с Яндекс.Диска.

Использование:
    python data/download.py
    python data/download.py --output data/raw
    python data/download.py --files ozon_train.csv image_embeddings.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve


YANDEX_DISK_PUBLIC_URL = "https://disk.yandex.ru/d/aw6epg3MNkQ9vw"
API_BASE = "https://cloud-api.yandex.net/v1/disk/public/resources"

DEFAULT_FILES = [
    "ozon_train.csv",
    "image_embeddings.csv",
    "clip_embeddings.parquet",
]

EXPECTED_SIZES_MB = {
    "ozon_train.csv": 191,
    "image_embeddings.csv": 91,
    "clip_embeddings.parquet": 592,
}


def get_download_link(public_url: str, file_path: str = "") -> str:
    """Получить прямую ссылку на скачивание файла с публичного Я.Диска."""
    api_url = (
        f"{API_BASE}/download?public_key={quote(public_url)}"
        + (f"&path={quote('/' + file_path)}" if file_path else "")
    )
    with urlopen(api_url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["href"]


def list_files(public_url: str) -> list[str]:
    """Получить список файлов в публичной папке Я.Диска."""
    api_url = f"{API_BASE}?public_key={quote(public_url)}&limit=100"
    with urlopen(api_url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    items = data.get("_embedded", {}).get("items", [])
    return [item["name"] for item in items if item.get("type") == "file"]


def download_file(url: str, dest: Path, expected_mb: float | None = None) -> None:
    """Скачать файл с прогрессом."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def progress(blocks: int, block_size: int, total_size: int) -> None:
        downloaded = blocks * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            sys.stdout.write(f"\r  [{pct:>3}%] {mb:>6.1f} / {total_mb:.1f} MB")
            sys.stdout.flush()

    urlretrieve(url, dest, reporthook=progress)
    print()

    actual_mb = dest.stat().st_size / 1024 / 1024
    if expected_mb and abs(actual_mb - expected_mb) > expected_mb * 0.15:
        print(
            f"  ⚠️  Размер {actual_mb:.1f} MB отличается от ожидаемого {expected_mb} MB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Папка назначения (по умолчанию data/raw)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Файлы для скачивания (по умолчанию все)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Только показать список файлов на Я.Диске",
    )
    args = parser.parse_args()

    if args.list:
        print(f"Файлы в {YANDEX_DISK_PUBLIC_URL}:")
        for name in list_files(YANDEX_DISK_PUBLIC_URL):
            print(f"  {name}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Папка назначения: {args.output.resolve()}")
    print(f"Файлы для скачивания: {', '.join(args.files)}")
    print()

    for filename in args.files:
        dest = args.output / filename
        if dest.exists():
            mb = dest.stat().st_size / 1024 / 1024
            print(f"✓ {filename} уже существует ({mb:.1f} MB), пропускаю")
            continue

        print(f"⏳ {filename}")
        try:
            url = get_download_link(YANDEX_DISK_PUBLIC_URL, filename)
            download_file(url, dest, EXPECTED_SIZES_MB.get(filename))
            print(f"✓ {filename} сохранён в {dest}")
        except Exception as exc:
            print(f"❌ Ошибка при скачивании {filename}: {exc}")
            sys.exit(1)

    print()
    print("Готово. Все файлы доступны в", args.output.resolve())


if __name__ == "__main__":
    main()
