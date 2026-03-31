"""
文件与资源操作封装函数。

支持图片和文件的上传/下载，返回可序列化的 dict。

示例：
    from scripts.files import upload_image, upload_file, download_resource
    import asyncio

    image_key = asyncio.run(upload_image("/tmp/photo.png"))
    file_key = asyncio.run(upload_file("/tmp/report.pdf", file_type="pdf"))
    saved = asyncio.run(download_resource(file_key, "/tmp/downloaded.pdf"))
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from .factory import get_platform


# ---------------------------------------------------------------------------
# 上传
# ---------------------------------------------------------------------------

_MIME_TO_FEISHU_TYPE = {
    "audio/ogg": "opus",
    "video/mp4": "mp4",
    "application/pdf": "pdf",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "doc",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xls",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "ppt",
}

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}


def _guess_feishu_file_type(file_path: str) -> str:
    """根据文件路径猜测飞书文件类型字符串。"""
    # suffix = Path(file_path).suffix.lower()
    mime, _ = mimetypes.guess_type(file_path)
    if mime in _MIME_TO_FEISHU_TYPE:
        return _MIME_TO_FEISHU_TYPE[mime]
    return "stream"


async def upload_image(
    file_path: str,
    platform: str | None = None,
) -> dict:
    """
    上传图片到 IM 平台，返回 image_key。

    上传成功后，可在发送消息时使用 image_key 引用该图片。

    Args:
        file_path: 本地图片路径（支持 jpg/png/gif/webp 等格式）
        platform: 指定平台（None 时从 IM_PLATFORM 环境变量读取）

    Returns:
        dict，含 image_key 和 file_path 字段

    示例：
        result = await upload_image("/tmp/chart.png")
        image_key = result["image_key"]
        # 之后发送图片消息：
        # await send_message(receive_id, json.dumps({"image_key": image_key}), msg_type="image")
    """
    p = get_platform(platform)
    image_key = await p.upload_image(file_path)
    return {"image_key": image_key, "file_path": file_path}


async def upload_file(
    file_path: str,
    file_type: str | None = None,
    platform: str | None = None,
) -> dict:
    """
    上传文件到 IM 平台，返回 file_key。

    上传成功后，可在发送消息时使用 file_key 引用该文件。

    Args:
        file_path: 本地文件路径
        file_type: 飞书文件类型，opus | mp4 | pdf | doc | xls | ppt | stream。
                   为 None 时根据扩展名自动推断（推断不出则使用 stream）
        platform: 指定平台

    Returns:
        dict，含 file_key、file_name、file_type、file_path 字段

    示例：
        result = await upload_file("/tmp/report.pdf")
        file_key = result["file_key"]
    """
    p = get_platform(platform)
    resolved_type = file_type or _guess_feishu_file_type(file_path)
    file_key = await p.upload_file(file_path, file_type=resolved_type)
    return {
        "file_key": file_key,
        "file_name": Path(file_path).name,
        "file_type": resolved_type,
        "file_path": file_path,
    }


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------

async def download_resource(
    file_key: str,
    save_path: str,
    platform: str | None = None,
) -> dict:
    """
    下载 IM 平台的文件或图片资源到本地。

    适用于从消息中获取到的 image_key 或 file_key。

    Args:
        file_key: 资源 key（image_key 或 file_key）
        save_path: 本地保存路径（含文件名）
        platform: 指定平台

    Returns:
        dict，含 saved_path 和 file_key 字段

    示例：
        # 先从消息中获取 image_key，再下载
        result = await download_resource("img_xxx", "/tmp/downloaded.png")
        print(result["saved_path"])
    """
    p = get_platform(platform)
    saved = await p.download_file(file_key, save_path)
    return {"saved_path": saved, "file_key": file_key}
