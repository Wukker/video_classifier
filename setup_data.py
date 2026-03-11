#!/usr/bin/env python3
"""
YT8M Dataset Setup Script
Создаёт папку data2, скачивает download.py и все tfrecord файлы (shard=1,10)
Использование: python setup_data.py
"""
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

BASE_DIR      = Path(__file__).parent / 'data2'
VALIDATE_DIR  = BASE_DIR / 'video' / 'validate'
DOWNLOAD_PY   = BASE_DIR / 'download.py'
DOWNLOAD_URL  = 'http://data.yt8m.org/download.py'
VOCAB_URL     = 'https://raw.githubusercontent.com/yash1994/youtube-8m-videos-downloader/master/vocabulary.csv'
VOCAB_PATH    = BASE_DIR / 'vocabulary_full.csv'

def run(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Ошибка: {result.stderr[-300:]}")
        sys.exit(1)
    return result

def main():
    print("=" * 55)
    print("  YT8M Dataset Setup")
    print("=" * 55)

    # 1. Создаём папки
    print("\n📂 Создаём папки...")
    VALIDATE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / 'processed').mkdir(exist_ok=True)
    (BASE_DIR / 'eda').mkdir(exist_ok=True)
    (BASE_DIR / 'stage3').mkdir(exist_ok=True)
    (BASE_DIR / 'stage4').mkdir(exist_ok=True)
    (BASE_DIR / 'stage5').mkdir(exist_ok=True)
    (BASE_DIR / 'inference').mkdir(exist_ok=True)
    print(f"   ✅ {BASE_DIR}")

    # 2. Скачиваем download.py
    print("\n📥 Скачиваем download.py...", end=" ", flush=True)
    urllib.request.urlretrieve(DOWNLOAD_URL, DOWNLOAD_PY)
    print("✅")

    # 3. Скачиваем словарь меток
    print("📥 Скачиваем vocabulary_full.csv...", end=" ", flush=True)
    urllib.request.urlretrieve(VOCAB_URL, VOCAB_PATH)
    print("✅")

    # 4. Считаем уже скачанные tfrecord
    existing = list(VALIDATE_DIR.glob("*.tfrecord"))
    print(f"\n📊 Уже скачано: {len(existing)} tfrecord файлов")

    if len(existing) >= 363:
        print("✅ Все файлы уже скачаны!")
        return

    # 5. Скачиваем tfrecord (shard=1,10 → ~380 файлов, ~540 MB)
    print(f"\n🔄 Скачиваем tfrecord файлы (shard=1,10 ≈ 380 файлов, ~540 MB)...")
    print("   Это займёт 5-15 минут в зависимости от скорости интернета\n")

    env = os.environ.copy()
    env['partition'] = '2/video/validate'
    env['mirror']    = 'eu'
    env['shard']     = '1,10'

    result = subprocess.run(
        [sys.executable, str(DOWNLOAD_PY)],
        cwd=str(VALIDATE_DIR),
        env=env,
    )

    # 6. Итог
    downloaded = list(VALIDATE_DIR.glob("*.tfrecord"))
    print(f"\n{'=' * 55}")
    print(f"  ГОТОВО")
    print(f"{'=' * 55}")
    print(f"  tfrecord файлов : {len(downloaded)}")
    total_mb = sum(f.stat().st_size for f in downloaded) / 1024**2
    print(f"  Размер на диске : {total_mb:.0f} MB")
    print(f"  Папка           : {VALIDATE_DIR}")
    print(f"{'=' * 55}")
    print("  Следующий шаг: открыть yt8m_classifier.ipynb → Stage 1")

if __name__ == "__main__":
    main()