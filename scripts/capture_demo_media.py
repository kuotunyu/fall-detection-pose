"""Capture and validate the local Gradio demo with Playwright."""

from __future__ import annotations

import argparse
from pathlib import Path

from fall_detection.app.media_capture import parse_viewport


def capture(
    *,
    url: str,
    out_path: Path,
    viewport: dict[str, int],
    video: Path | None = None,
    timeout_ms: int = 180_000,
) -> None:
    from playwright.sync_api import sync_playwright

    console_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport=viewport)
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.goto(url, wait_until="domcontentloaded")
        page.locator("#fd-input").first.wait_for(state="visible")
        if video is not None:
            page.locator("#fd-input input[type=file]").set_input_files(str(video.resolve()))
            page.get_by_role("button", name="開始分析").click()
            page.locator("#fd-processing").first.wait_for(state="visible")
            page.locator("#fd-result").first.wait_for(
                state="visible", timeout=timeout_ms
            )
        overflow = page.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        )
        if overflow:
            raise AssertionError("page has horizontal overflow")
        actionable_errors = [
            message for message in console_errors if "favicon" not in message.lower()
        ]
        if actionable_errors:
            raise AssertionError(f"browser console errors: {actionable_errors}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(out_path), full_page=True)
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7863")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--viewport", default="1440x1000")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--timeout-ms", type=int, default=180_000)
    args = parser.parse_args()
    capture(
        url=args.url,
        out_path=args.out,
        viewport=parse_viewport(args.viewport),
        video=args.video,
        timeout_ms=args.timeout_ms,
    )


if __name__ == "__main__":
    main()
