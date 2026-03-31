"""
Media upload and sending functionality for Weixin SDK.

This module provides high-level methods for sending multimedia messages
(images, videos, files, voice) similar to TypeScript's send-media.ts.
"""

import base64
import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import aiohttp

from .types import (
    MessageItem,
    MessageItemType,
    ImageItem,
    VideoItem,
    FileItem,
    VoiceItem,
    CDNMedia,
    GetUploadUrlReq,
    UploadMediaType,
)
from .exceptions import WeixinError, WeixinAPIError

logger = logging.getLogger(__name__)


class MediaUploader:
    """
    Helper class for uploading media files to Weixin CDN.

    Similar to TypeScript's cdn-upload.ts functionality.
    """

    def __init__(self, client):
        """
        Initialize media uploader.

        Args:
            client: WeixinClient instance
        """
        self.client = client

    async def upload_file(
        self,
        file_path: str,
        to_user_id: str,
        media_type: int = UploadMediaType.FILE,
    ) -> Dict[str, Any]:
        """
        Upload a file to Weixin CDN.

        Args:
            file_path: Path to the file to upload
            to_user_id: Target user ID
            media_type: Type of media (IMAGE, VIDEO, FILE, VOICE)

        Returns:
            Dict containing uploaded file info:
            - filekey: File key
            - aeskey: AES encryption key
            - file_size: File size
            - download_param: Download parameter

        Raises:
            WeixinError: If upload fails
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise WeixinError(f"File not found: {path_obj}")

        # Read file
        file_content = path_obj.read_bytes()
        file_size = len(file_content)

        # Calculate MD5
        raw_md5 = hashlib.md5(file_content).hexdigest()

        # Generate AES key (16 bytes for AES-128)
        aes_key = os.urandom(16)

        # Encrypt file content (AES-128-ECB)
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad

        cipher = AES.new(aes_key, AES.MODE_ECB)
        encrypted_content = cipher.encrypt(pad(file_content, AES.block_size))
        encrypted_size = len(encrypted_content)

        # Get upload URL
        filekey = str(uuid.uuid4())
        req = GetUploadUrlReq(
            filekey=filekey,
            media_type=media_type,
            to_user_id=to_user_id,
            rawsize=file_size,
            rawfilemd5=raw_md5,
            filesize=encrypted_size,
            aeskey=base64.b64encode(aes_key).decode(),
        )

        upload_url_resp = await self.client.get_upload_url(req)

        if not upload_url_resp.upload_param:
            raise WeixinError("Failed to get upload URL")

        # Upload encrypted content to CDN
        async with aiohttp.ClientSession() as session:
            async with session.post(
                upload_url_resp.upload_param,
                data=encrypted_content,
                headers={"Content-Type": "application/octet-stream"},
            ) as response:
                if response.status != 200:
                    raise WeixinAPIError(
                        f"CDN upload failed: {response.status}",
                        code=response.status,
                    )

        logger.info(f"Uploaded file {filekey} ({file_size} bytes)")

        return {
            "filekey": filekey,
            "aeskey": aes_key,
            "file_size": file_size,
            "file_size_ciphertext": encrypted_size,
            "download_param": upload_url_resp.upload_param,
        }

    async def upload_image(self, image_path: str, to_user_id: str) -> Dict[str, Any]:
        """Upload an image file."""
        return await self.upload_file(
            image_path,
            to_user_id,
            media_type=UploadMediaType.IMAGE,
        )

    async def upload_video(self, video_path: str, to_user_id: str) -> Dict[str, Any]:
        """Upload a video file."""
        return await self.upload_file(
            video_path,
            to_user_id,
            media_type=UploadMediaType.VIDEO,
        )


class MediaSender:
    """
    Helper class for sending media messages.

    Similar to TypeScript's send-media.ts functionality.
    """

    def __init__(self, client):
        """
        Initialize media sender.

        Args:
            client: WeixinClient instance
        """
        self.client = client
        self.uploader = MediaUploader(client)

    async def send_image(
        self,
        to_user_id: str,
        image_path: str,
        caption: str = "",
        context_token: Optional[str] = None,
    ) -> str:
        """
        Send an image message.

        Args:
            to_user_id: Target user ID
            image_path: Path to image file
            caption: Optional text caption
            context_token: Optional context token

        Returns:
            Message ID

        Example:
            >>> await sender.send_image(
            ...     to_user_id="user@im.wechat",
            ...     image_path="/path/to/photo.jpg",
            ...     caption="Check this out!"
            ... )
        """
        # Upload image
        uploaded = await self.uploader.upload_image(image_path, to_user_id)

        # Build image item
        image_item = MessageItem(
            type=MessageItemType.IMAGE,
            image_item=ImageItem(
                media=CDNMedia(
                    encrypt_query_param=uploaded["download_param"],
                    aes_key=base64.b64encode(uploaded["aeskey"]).decode(),
                    encrypt_type=1,
                ),
                mid_size=uploaded["file_size_ciphertext"],
            ),
        )

        # Send message
        return await self._send_media_items(
            to_user_id=to_user_id,
            items=[image_item],
            caption=caption,
            context_token=context_token,
        )

    async def send_video(
        self,
        to_user_id: str,
        video_path: str,
        caption: str = "",
        context_token: Optional[str] = None,
    ) -> str:
        """
        Send a video message.

        Args:
            to_user_id: Target user ID
            video_path: Path to video file
            caption: Optional text caption
            context_token: Optional context token

        Returns:
            Message ID
        """
        # Upload video
        uploaded = await self.uploader.upload_video(video_path, to_user_id)

        # Build video item
        video_item = MessageItem(
            type=MessageItemType.VIDEO,
            video_item=VideoItem(
                media=CDNMedia(
                    encrypt_query_param=uploaded["download_param"],
                    aes_key=base64.b64encode(uploaded["aeskey"]).decode(),
                    encrypt_type=1,
                ),
                video_size=uploaded["file_size_ciphertext"],
            ),
        )

        # Send message
        return await self._send_media_items(
            to_user_id=to_user_id,
            items=[video_item],
            caption=caption,
            context_token=context_token,
        )

    async def send_file(
        self,
        to_user_id: str,
        file_path: str,
        caption: str = "",
        context_token: Optional[str] = None,
    ) -> str:
        """
        Send a file attachment.

        Args:
            to_user_id: Target user ID
            file_path: Path to file
            caption: Optional text caption
            context_token: Optional context token

        Returns:
            Message ID
        """
        from pathlib import Path

        file_path_obj = Path(file_path)
        file_name = file_path_obj.name

        # Upload file
        uploaded = await self.uploader.upload_file(
            file_path,
            to_user_id,
            media_type=UploadMediaType.FILE,
        )

        # Calculate file MD5 for FileItem
        file_content = file_path_obj.read_bytes()
        file_md5 = hashlib.md5(file_content).hexdigest()

        # Build file item
        file_item = MessageItem(
            type=MessageItemType.FILE,
            file_item=FileItem(
                media=CDNMedia(
                    encrypt_query_param=uploaded["download_param"],
                    aes_key=base64.b64encode(uploaded["aeskey"]).decode(),
                    encrypt_type=1,
                ),
                file_name=file_name,
                md5=file_md5,
                len=str(uploaded["file_size"]),
            ),
        )

        # Send message
        return await self._send_media_items(
            to_user_id=to_user_id,
            items=[file_item],
            caption=caption,
            context_token=context_token,
        )

    async def send_voice(
        self,
        to_user_id: str,
        voice_path: str,
        duration_ms: int = 0,
        context_token: Optional[str] = None,
    ) -> str:
        """
        Send a voice message.

        Args:
            to_user_id: Target user ID
            voice_path: Path to voice file (should be in SILK format)
            duration_ms: Voice duration in milliseconds
            context_token: Optional context token

        Returns:
            Message ID

        Note:
            For best compatibility, voice files should be in SILK format.
            You can use silk-wasm or similar tools to convert from MP3/WAV.
        """
        # Upload voice
        uploaded = await self.uploader.upload_file(
            voice_path,
            to_user_id,
            media_type=UploadMediaType.VOICE,
        )

        # Build voice item
        voice_item = MessageItem(
            type=MessageItemType.VOICE,
            voice_item=VoiceItem(
                media=CDNMedia(
                    encrypt_query_param=uploaded["download_param"],
                    aes_key=base64.b64encode(uploaded["aeskey"]).decode(),
                    encrypt_type=1,
                ),
                encode_type=6,  # SILK format
                playtime=duration_ms,
            ),
        )

        # Send message
        return await self._send_media_items(
            to_user_id=to_user_id,
            items=[voice_item],
            caption="",  # Voice messages don't have captions
            context_token=context_token,
        )

    async def _send_media_items(
        self,
        to_user_id: str,
        items: list,
        caption: str = "",
        context_token: Optional[str] = None,
    ) -> str:
        """
        Send one or more media items with optional caption.

        Args:
            to_user_id: Target user ID
            items: List of MessageItem objects
            caption: Optional text caption (sent as separate TEXT item first)
            context_token: Optional context token

        Returns:
            Last message ID
        """
        from .types import WeixinMessage, MessageType, MessageState, TextItem
        from .types import SendMessageReq

        last_client_id = ""

        # Send caption as text item if provided
        if caption:
            last_client_id = str(uuid.uuid4())
            text_msg = WeixinMessage(
                to_user_id=to_user_id,
                client_id=last_client_id,
                message_type=MessageType.BOT,
                message_state=MessageState.FINISH,
                item_list=[
                    MessageItem(
                        type=MessageItemType.TEXT,
                        text_item=TextItem(text=caption),
                    )
                ],
                context_token=context_token,
            )
            await self.client.send_message(SendMessageReq(msg=text_msg))

        # Send each media item
        for item in items:
            last_client_id = str(uuid.uuid4())
            media_msg = WeixinMessage(
                to_user_id=to_user_id,
                client_id=last_client_id,
                message_type=MessageType.BOT,
                message_state=MessageState.FINISH,
                item_list=[item],
                context_token=context_token,
            )
            await self.client.send_message(SendMessageReq(msg=media_msg))

        return last_client_id


# Convenience functions for standalone usage


async def send_image(
    client,
    to_user_id: str,
    image_path: str,
    caption: str = "",
    context_token: Optional[str] = None,
) -> str:
    """Standalone function to send image."""
    sender = MediaSender(client)
    return await sender.send_image(to_user_id, image_path, caption, context_token)


async def send_video(
    client,
    to_user_id: str,
    video_path: str,
    caption: str = "",
    context_token: Optional[str] = None,
) -> str:
    """Standalone function to send video."""
    sender = MediaSender(client)
    return await sender.send_video(to_user_id, video_path, caption, context_token)


async def send_file(
    client,
    to_user_id: str,
    file_path: str,
    caption: str = "",
    context_token: Optional[str] = None,
) -> str:
    """Standalone function to send file."""
    sender = MediaSender(client)
    return await sender.send_file(to_user_id, file_path, caption, context_token)


async def send_voice(
    client,
    to_user_id: str,
    voice_path: str,
    duration_ms: int = 0,
    context_token: Optional[str] = None,
) -> str:
    """Standalone function to send voice."""
    sender = MediaSender(client)
    return await sender.send_voice(to_user_id, voice_path, duration_ms, context_token)
