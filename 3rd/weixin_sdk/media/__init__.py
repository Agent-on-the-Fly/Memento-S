"""
WeChat media handling module.

Provides transcoding capabilities for WeChat voice messages (SILK format),
media upload, and media sending functionality.
"""

# Import from media_upload.py (in parent directory)
from weixin_sdk.media_upload import (
    MediaUploader,
    MediaSender,
    send_image,
    send_video,
    send_file,
    send_voice,
)

# Import from transcode module
from weixin_sdk.media.transcode import (
    SilkTranscoder,
    TranscodeBackend,
    quick_transcode,
)

__all__ = [
    # Transcoding
    "SilkTranscoder",
    "TranscodeBackend",
    "quick_transcode",
    # Media upload/send
    "MediaUploader",
    "MediaSender",
    "send_image",
    "send_video",
    "send_file",
    "send_voice",
]
