"""
Weixin protocol types (mirrors TypeScript types from src/api/types.ts)
API uses JSON over HTTP; bytes fields are base64 strings in JSON.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import IntEnum


class MessageType(IntEnum):
    """Message type enumeration."""

    NONE = 0
    USER = 1
    BOT = 2


class MessageItemType(IntEnum):
    """Message item type enumeration."""

    NONE = 0
    TEXT = 1
    IMAGE = 2
    VOICE = 3
    FILE = 4
    VIDEO = 5


class MessageState(IntEnum):
    """Message state enumeration."""

    NEW = 0
    GENERATING = 1
    FINISH = 2


class UploadMediaType(IntEnum):
    """Upload media type enumeration."""

    IMAGE = 1
    VIDEO = 2
    FILE = 3
    VOICE = 4


class TypingStatus(IntEnum):
    """Typing status enumeration."""

    TYPING = 1
    CANCEL = 2


@dataclass
class BaseInfo:
    """Common request metadata attached to every CGI request."""

    channel_version: Optional[str] = None


@dataclass
class TextItem:
    """Text message item."""

    text: Optional[str] = None


@dataclass
class CDNMedia:
    """CDN media reference; aes_key is base64-encoded bytes in JSON."""

    encrypt_query_param: Optional[str] = None
    aes_key: Optional[str] = None
    encrypt_type: Optional[int] = None  # 0=只加密fileid, 1=打包缩略图/中图等信息


@dataclass
class ImageItem:
    """Image message item."""

    media: Optional[CDNMedia] = None
    thumb_media: Optional[CDNMedia] = None
    aeskey: Optional[str] = None  # Raw AES-128 key as hex string (16 bytes)
    url: Optional[str] = None
    mid_size: Optional[int] = None
    thumb_size: Optional[int] = None
    thumb_height: Optional[int] = None
    thumb_width: Optional[int] = None
    hd_size: Optional[int] = None


@dataclass
class VoiceItem:
    """Voice message item."""

    media: Optional[CDNMedia] = None
    encode_type: Optional[int] = (
        None  # 1=pcm 2=adpcm 3=feature 4=speex 5=amr 6=silk 7=mp3 8=ogg-speex
    )
    bits_per_sample: Optional[int] = None
    sample_rate: Optional[int] = None  # Hz
    playtime: Optional[int] = None  # milliseconds
    text: Optional[str] = None  # 语音转文字内容


@dataclass
class FileItem:
    """File message item."""

    media: Optional[CDNMedia] = None
    file_name: Optional[str] = None
    md5: Optional[str] = None
    len: Optional[str] = None


@dataclass
class VideoItem:
    """Video message item."""

    media: Optional[CDNMedia] = None
    video_size: Optional[int] = None
    play_length: Optional[int] = None
    video_md5: Optional[str] = None
    thumb_media: Optional[CDNMedia] = None
    thumb_size: Optional[int] = None
    thumb_height: Optional[int] = None
    thumb_width: Optional[int] = None


@dataclass
class RefMessage:
    """Referenced message."""

    message_item: Optional["MessageItem"] = None
    title: Optional[str] = None  # 摘要


@dataclass
class MessageItem:
    """Message item (text/image/voice/file/video)."""

    type: Optional[int] = None
    create_time_ms: Optional[int] = None
    update_time_ms: Optional[int] = None
    is_completed: Optional[bool] = None
    msg_id: Optional[str] = None
    ref_msg: Optional[RefMessage] = None
    text_item: Optional[TextItem] = None
    image_item: Optional[ImageItem] = None
    voice_item: Optional[VoiceItem] = None
    file_item: Optional[FileItem] = None
    video_item: Optional[VideoItem] = None


@dataclass
class WeixinMessage:
    """Unified message structure."""

    seq: Optional[int] = None
    message_id: Optional[int] = None
    from_user_id: Optional[str] = None
    to_user_id: Optional[str] = None
    client_id: Optional[str] = None
    create_time_ms: Optional[int] = None
    update_time_ms: Optional[int] = None
    delete_time_ms: Optional[int] = None
    session_id: Optional[str] = None
    group_id: Optional[str] = None
    message_type: Optional[int] = None
    message_state: Optional[int] = None
    item_list: Optional[List[MessageItem]] = field(default_factory=list)
    context_token: Optional[str] = None


@dataclass
class GetUpdatesReq:
    """GetUpdates request."""

    sync_buf: Optional[str] = None  # @deprecated compat only
    get_updates_buf: Optional[str] = None  # Full context buf cached locally


@dataclass
class GetUpdatesResp:
    """GetUpdates response."""

    ret: Optional[int] = None
    errcode: Optional[int] = None  # e.g. -14 = session timeout
    errmsg: Optional[str] = None
    msgs: Optional[List[WeixinMessage]] = field(default_factory=list)
    sync_buf: Optional[str] = None  # @deprecated compat only
    get_updates_buf: Optional[str] = None  # Full context buf to cache
    longpolling_timeout_ms: Optional[int] = None


@dataclass
class SendMessageReq:
    """SendMessage request."""

    msg: Optional[WeixinMessage] = None


@dataclass
class SendMessageResp:
    """SendMessage response."""

    pass


@dataclass
class SendTypingReq:
    """SendTyping request."""

    ilink_user_id: Optional[str] = None
    typing_ticket: Optional[str] = None
    status: Optional[int] = None  # 1=typing (default), 2=cancel typing


@dataclass
class SendTypingResp:
    """SendTyping response."""

    ret: Optional[int] = None
    errmsg: Optional[str] = None


@dataclass
class GetConfigResp:
    """GetConfig response."""

    ret: Optional[int] = None
    errmsg: Optional[str] = None
    typing_ticket: Optional[str] = None  # Base64-encoded typing ticket


@dataclass
class GetUploadUrlReq:
    """GetUploadUrl request."""

    filekey: Optional[str] = None
    media_type: Optional[int] = None
    to_user_id: Optional[str] = None
    rawsize: Optional[int] = None
    rawfilemd5: Optional[str] = None
    filesize: Optional[int] = None
    thumb_rawsize: Optional[int] = None
    thumb_rawfilemd5: Optional[str] = None
    thumb_filesize: Optional[int] = None
    no_need_thumb: Optional[bool] = None
    aeskey: Optional[str] = None


@dataclass
class GetUploadUrlResp:
    """GetUploadUrl response."""

    upload_param: Optional[str] = None
    thumb_upload_param: Optional[str] = None


# Helper functions for converting between dict and dataclass


def dict_to_message_item(data: Dict[str, Any]) -> MessageItem:
    """Convert dict to MessageItem."""
    return MessageItem(
        type=data.get("type"),
        create_time_ms=data.get("create_time_ms"),
        update_time_ms=data.get("update_time_ms"),
        is_completed=data.get("is_completed"),
        msg_id=data.get("msg_id"),
        ref_msg=RefMessage(**data["ref_msg"]) if data.get("ref_msg") else None,
        text_item=TextItem(**data["text_item"]) if data.get("text_item") else None,
        image_item=ImageItem(**data["image_item"]) if data.get("image_item") else None,
        voice_item=VoiceItem(**data["voice_item"]) if data.get("voice_item") else None,
        file_item=FileItem(**data["file_item"]) if data.get("file_item") else None,
        video_item=VideoItem(**data["video_item"]) if data.get("video_item") else None,
    )


def dict_to_weixin_message(data: Dict[str, Any]) -> WeixinMessage:
    """Convert dict to WeixinMessage."""
    return WeixinMessage(
        seq=data.get("seq"),
        message_id=data.get("message_id"),
        from_user_id=data.get("from_user_id"),
        to_user_id=data.get("to_user_id"),
        client_id=data.get("client_id"),
        create_time_ms=data.get("create_time_ms"),
        update_time_ms=data.get("update_time_ms"),
        delete_time_ms=data.get("delete_time_ms"),
        session_id=data.get("session_id"),
        group_id=data.get("group_id"),
        message_type=data.get("message_type"),
        message_state=data.get("message_state"),
        item_list=[dict_to_message_item(item) for item in data.get("item_list", [])],
        context_token=data.get("context_token"),
    )


def dict_to_get_updates_resp(data: Dict[str, Any]) -> GetUpdatesResp:
    """Convert dict to GetUpdatesResp."""
    return GetUpdatesResp(
        ret=data.get("ret"),
        errcode=data.get("errcode"),
        errmsg=data.get("errmsg"),
        msgs=[dict_to_weixin_message(msg) for msg in data.get("msgs", [])],
        sync_buf=data.get("sync_buf"),
        get_updates_buf=data.get("get_updates_buf"),
        longpolling_timeout_ms=data.get("longpolling_timeout_ms"),
    )
