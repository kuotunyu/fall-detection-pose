"""UR Fall Detection Dataset(URFD)的下載與前處理。

官方站(2026-07 確認存活):https://fenix.ur.edu.pl/~mkepski/ds/uf.html
(舊網域 fenix.univ.rzeszow.pl 已失效,許多論文/舊腳本裡的連結不可用。)

- 資料為每序列一個 PNG zip(cam0 = 平行地面側視;ADL 只有 cam0),
  640x480 @ 30fps;本專案只用 cam0 RGB。
- 無打包下載:70 個 zip 逐檔抓。官方站是小型大學伺服器,
  下載間隔與重試皆放禮貌值,且一律 skip-existing(Colab 斷線重跑不重工)。
- 授權 CC BY-NC-SA 4.0:資料不進 git、不重新上傳;引用見 README。
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np

BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"
ANNOTATION_URLS = {
    "falls": f"{BASE_URL}/urfall-cam0-falls.csv",
    "adls": f"{BASE_URL}/urfall-cam0-adls.csv",
}
URFD_FPS = 30.0


def fall_sequences() -> list[str]:
    return [f"fall-{i:02d}" for i in range(1, 31)]


def adl_sequences() -> list[str]:
    return [f"adl-{i:02d}" for i in range(1, 41)]


def all_sequences() -> list[str]:
    return fall_sequences() + adl_sequences()


def rgb_zip_url(sequence: str) -> str:
    return f"{BASE_URL}/{sequence}-cam0-rgb.zip"


def download_file(
    url: str,
    dst: str | Path,
    retries: int = 4,
    backoff_s: float = 3.0,
    timeout_s: float = 120.0,
) -> Path:
    """單檔下載:已存在且非空即跳過;先寫 .part 再改名,中斷不留半成品。"""
    import requests

    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout_s) as resp:
                resp.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            part.replace(dst)
            return dst
        except Exception as e:  # noqa: BLE001 - 重試涵蓋網路層各種錯誤
            last_err = e
            time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"下載失敗(重試 {retries} 次):{url}") from last_err


def download_annotations(data_dir: str | Path) -> dict[str, Path]:
    """下載兩份標註 CSV,回傳 {'falls': path, 'adls': path}。"""
    data_dir = Path(data_dir)
    return {
        key: download_file(url, data_dir / Path(url).name)
        for key, url in ANNOTATION_URLS.items()
    }


def download_sequences(
    data_dir: str | Path,
    sequences: list[str] | None = None,
    polite_sleep_s: float = 0.5,
    progress: bool = True,
) -> list[Path]:
    """下載指定序列的 cam0 RGB zip(預設全部 70 個);一律 skip-existing。"""
    data_dir = Path(data_dir)
    sequences = sequences if sequences is not None else all_sequences()
    out = []
    for i, seq in enumerate(sequences):
        dst = data_dir / "zips" / f"{seq}-cam0-rgb.zip"
        already = dst.exists() and dst.stat().st_size > 0
        out.append(download_file(rgb_zip_url(seq), dst))
        if progress:
            size_mb = dst.stat().st_size / 1e6
            status = "skip" if already else "done"
            print(f"[{i + 1}/{len(sequences)}] {seq}: {status} ({size_mb:.1f} MB)")
        if not already and polite_sleep_s > 0:
            time.sleep(polite_sleep_s)  # 小型大學伺服器,禮貌間隔
    return out


def zip_to_video(zip_path: str | Path, out_path: str | Path, fps: float = URFD_FPS) -> Path:
    """PNG 序列 zip → mp4(幀序依檔名排序;輸出已存在即跳過)。

    重組出的 mp4 僅供推論讀取,用 mp4v 即可(非瀏覽器播放用途)。
    """
    import cv2

    from .video import write_video_mp4v

    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    def frames():
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith(".png"))
            if not names:
                raise RuntimeError(f"{zip_path} 內沒有 PNG")
            for name in names:
                buf = np.frombuffer(zf.read(name), dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"{zip_path}:{name} 解碼失敗")
                yield frame

    tmp = out_path.with_suffix(".tmp.mp4")
    n = write_video_mp4v(frames(), tmp, fps)
    tmp.replace(out_path)
    print(f"{Path(zip_path).name} → {out_path.name} ({n} 幀)")
    return out_path


def build_videos(
    data_dir: str | Path, sequences: list[str] | None = None, fps: float = URFD_FPS
) -> dict[str, Path]:
    """批次把已下載的 zip 重組成 mp4;回傳 {sequence: 影片路徑}。"""
    data_dir = Path(data_dir)
    sequences = sequences if sequences is not None else all_sequences()
    out: dict[str, Path] = {}
    for seq in sequences:
        zip_path = data_dir / "zips" / f"{seq}-cam0-rgb.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"缺少 {zip_path},請先執行 download_sequences")
        out[seq] = zip_to_video(zip_path, data_dir / "videos" / f"{seq}.mp4", fps)
    return out
