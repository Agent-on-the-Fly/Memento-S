"""
Gateway 渠道适配器离线测试。

不启动任何项目服务即可运行：
    python -m pytest middleware/im/test_gateway_channels.py -v

测试覆盖：
  1. 消息格式转换（_convert_to_gateway_message / _parse_*）—— 纯函数，无网络
  2. Webhook payload 解析（parse_webhook）—— 无 HTTP 服务器
  3. 消息发送路由（send_message / reply_message）—— mock platform
  4. Webhook 签名校验（verify_webhook）
"""
from __future__ import annotations

import sys
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from middleware.im.gateway.protocol import (
    ChannelType,
    ConnectionConfig,
    ConnectionMode,
    ConnectionType,
    GatewayMessage,
    MessageType,
)
from middleware.im.im_platform.base import IMMessage


# ---------------------------------------------------------------------------
# 确保项目根目录在 sys.path
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# 工具：构造 ConnectionConfig
# ---------------------------------------------------------------------------


def _make_config(channel_type: ChannelType = ChannelType.FEISHU) -> ConnectionConfig:
    return ConnectionConfig(
        connection_id="test-conn",
        connection_type=ConnectionType.CHANNEL,
        channel_type=channel_type,
        channel_account="test_account",
        mode=ConnectionMode.WEBSOCKET,
    )


def _make_im_message(**kwargs) -> IMMessage:
    defaults = dict(
        id="msg_001",
        chat_id="chat_001",
        sender_id="bot",
        content="hello",
        msg_type="text",
        create_time="1700000000000",
    )
    defaults.update(kwargs)
    return IMMessage(**defaults)


# ===========================================================================
# 飞书 FeishuAdapter
# ===========================================================================

class TestFeishuAdapter:

    def _make_adapter(self):
        """构造适配器并注入 mock platform，跳过真实凭证检查。"""
        from middleware.im.gateway.channels.feishu import FeishuAdapter
        adapter = FeishuAdapter(
            app_id="test_app_id",
            app_secret="test_app_secret",
        )
        adapter._config = _make_config(ChannelType.FEISHU)
        # 注入 mock platform（跳过真实 API）
        adapter._platform = MagicMock()
        return adapter

    # ---- 消息格式转换 ----

    def test_convert_to_gateway_message_basic(self):
        adapter = self._make_adapter()
        msg_dict = {
            "id": "msg_001",
            "chat_id": "chat_abc",
            "sender_id": "user_xyz",
            "content": "你好",
            "msg_type": "text",
            "create_time": "1700000000000",
            "parent_id": "parent_001",
            "root_id": "root_001",
        }
        gw = adapter._convert_to_gateway_message(msg_dict)

        assert isinstance(gw, GatewayMessage)
        assert gw.chat_id == "chat_abc"
        assert gw.sender_id == "user_xyz"
        assert gw.content == "你好"
        assert gw.msg_type == "text"
        assert gw.reply_to == "parent_001"
        assert gw.thread_id == "root_001"
        assert gw.channel_type == ChannelType.FEISHU
        assert gw.metadata["message_id"] == "msg_001"

    def test_parse_event_to_dict(self):
        adapter = self._make_adapter()
        event = {
            "message": {
                "message_id": "om_abc",
                "chat_id": "oc_chat",
                "message_type": "text",
                "content": '{"text":"hello"}',
                "create_time": "1700000000",
                "parent_id": "om_parent",
                "root_id": "om_root",
            },
            "sender": {
                "sender_id": {"open_id": "ou_user"}
            },
        }
        result = adapter._parse_event_to_dict(event)

        assert result is not None
        assert result["id"] == "om_abc"
        assert result["chat_id"] == "oc_chat"
        assert result["sender_id"] == "ou_user"
        assert result["msg_type"] == "text"
        assert result["parent_id"] == "om_parent"
        assert result["root_id"] == "om_root"

    def test_parse_event_to_dict_no_message(self):
        adapter = self._make_adapter()
        result = adapter._parse_event_to_dict({})
        assert result is None

    # ---- Webhook 解析 ----

    @pytest.mark.asyncio
    async def test_parse_webhook_message_event(self):
        adapter = self._make_adapter()
        payload = {
            "event": {
                "message": {
                    "message_id": "om_001",
                    "chat_id": "oc_001",
                    "message_type": "text",
                    "content": '{"text":"test"}',
                    "create_time": "1700000000",
                    "parent_id": "",
                    "root_id": "",
                },
                "sender": {"sender_id": {"open_id": "ou_001"}},
            }
        }
        messages = await adapter.parse_webhook(payload)
        assert len(messages) == 1
        assert messages[0].chat_id == "oc_001"
        assert messages[0].sender_id == "ou_001"

    @pytest.mark.asyncio
    async def test_parse_webhook_url_verification(self):
        adapter = self._make_adapter()
        payload = {"type": "url_verification", "challenge": "abc123"}
        messages = await adapter.parse_webhook(payload)
        assert messages == []

    @pytest.mark.asyncio
    async def test_parse_webhook_no_event(self):
        adapter = self._make_adapter()
        messages = await adapter.parse_webhook({"other": "data"})
        assert messages == []

    # ---- 消息发送 ----

    @pytest.mark.asyncio
    async def test_send_message_with_chat_id(self):
        adapter = self._make_adapter()
        adapter._platform.send_message = AsyncMock(
            return_value=_make_im_message(id="new_msg")
        )
        result = await adapter.send_message("oc_chat", "hello", receive_id_type="chat_id")

        adapter._platform.send_message.assert_called_once_with(
            receive_id="oc_chat",
            content="hello",
            msg_type="text",
            receive_id_type="chat_id",
        )
        assert result == "new_msg"

    @pytest.mark.asyncio
    async def test_reply_message(self):
        adapter = self._make_adapter()
        adapter._platform.reply_message = AsyncMock(
            return_value=_make_im_message(id="reply_msg")
        )
        result = await adapter.reply_message("om_001", "reply content")

        adapter._platform.reply_message.assert_called_once_with(
            message_id="om_001",
            content="reply content",
            msg_type="text",
        )
        assert result == "reply_msg"

    # ---- Webhook 签名验证 ----

    @pytest.mark.asyncio
    async def test_verify_webhook_no_encrypt_key(self):
        adapter = self._make_adapter()
        adapter.encrypt_key = ""
        # 无 encrypt_key 时直接返回 True
        result = await adapter.verify_webhook("sig", b"body")
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_webhook_with_valid_signature(self):
        import hashlib
        import time

        adapter = self._make_adapter()
        adapter.encrypt_key = "test_encrypt_key"

        timestamp = str(int(time.time()))
        nonce = "random_nonce"
        body = b'{"event": {}}'
        sign_base = timestamp + nonce + adapter.encrypt_key + body.decode("utf-8")
        computed = hashlib.sha256(sign_base.encode()).hexdigest()

        headers = {
            "X-Lark-Request-Timestamp": timestamp,
            "X-Lark-Request-Nonce": nonce,
            "X-Lark-Signature": computed,
        }
        result = await adapter.verify_webhook(computed, body, headers=headers)
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_webhook_with_stale_timestamp(self):
        adapter = self._make_adapter()
        adapter.encrypt_key = "test_encrypt_key"
        headers = {
            "X-Lark-Request-Timestamp": "1000000",  # 很久以前
            "X-Lark-Request-Nonce": "nonce",
            "X-Lark-Signature": "anything",
        }
        result = await adapter.verify_webhook("anything", b"body", headers=headers)
        assert result is False


# ===========================================================================
# 钉钉 DingTalkAdapter
# ===========================================================================

class TestDingTalkAdapter:

    def _make_adapter(self):
        from middleware.im.gateway.channels.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter()
        adapter._config = _make_config(ChannelType.DINGTALK)
        adapter._platform = MagicMock()
        return adapter

    # ---- 消息格式转换 ----

    def test_convert_to_gateway_message(self):
        adapter = self._make_adapter()
        msg_dict = {
            "id": "dt_msg_001",
            "chat_id": "conv_001",
            "sender_id": "staff_001",
            "content": "钉钉消息",
            "msg_type": "text",
            "create_time": "1700000000000",
            "sender_nick": "张三",
            "chat_type": "group",
        }
        gw = adapter._convert_to_gateway_message(msg_dict)

        assert gw.chat_id == "conv_001"
        assert gw.sender_id == "staff_001"
        assert gw.content == "钉钉消息"
        assert gw.channel_type == ChannelType.DINGTALK
        assert gw.metadata["sender_nick"] == "张三"
        assert gw.metadata["chat_type"] == "group"

    def test_parse_robot_message_text(self):
        adapter = self._make_adapter()
        payload = {
            "msgtype": "text",
            "text": {"content": "hello robot"},
            "msgId": "msg_abc",
            "conversationId": "conv_abc",
            "senderStaffId": "staff_abc",
            "senderNick": "李四",
            "createAt": "1700000000",
            "conversationType": "2",
        }
        result = adapter._parse_robot_message(payload)

        assert result["id"] == "msg_abc"
        assert result["chat_id"] == "conv_abc"
        assert result["sender_id"] == "staff_abc"
        assert result["content"] == "hello robot"
        assert result["sender_nick"] == "李四"

    def test_parse_robot_message_markdown(self):
        adapter = self._make_adapter()
        payload = {
            "msgtype": "markdown",
            "markdown": {"text": "# Title\ncontent"},
            "msgId": "msg_md",
            "conversationId": "conv_md",
        }
        result = adapter._parse_robot_message(payload)
        assert result["content"] == "# Title\ncontent"

    def test_parse_stream_message(self):
        adapter = self._make_adapter()
        payload = {
            "msgId": "stream_001",
            "conversationId": "conv_stream",
            "senderId": "user_001",
            "content": {"content": "stream text"},
            "msgtype": "text",
            "createTime": "1700000000",
        }
        result = adapter._parse_stream_message(payload)
        assert result["id"] == "stream_001"
        assert result["content"] == "stream text"

    # ---- Webhook 解析 ----

    @pytest.mark.asyncio
    async def test_parse_webhook_robot_message(self):
        adapter = self._make_adapter()
        payload = {
            "msgtype": "text",
            "text": {"content": "hello"},
            "msgId": "msg_001",
            "conversationId": "conv_001",
            "senderStaffId": "staff_001",
        }
        messages = await adapter.parse_webhook(payload)
        assert len(messages) == 1
        assert messages[0].content == "hello"

    @pytest.mark.asyncio
    async def test_parse_webhook_stream_message(self):
        adapter = self._make_adapter()
        payload = {
            "conversationId": "conv_001",
            "senderId": "user_001",
            "content": {"content": "stream msg"},
            "msgId": "msg_001",
        }
        messages = await adapter.parse_webhook(payload)
        assert len(messages) == 1
        assert messages[0].content == "stream msg"

    @pytest.mark.asyncio
    async def test_parse_webhook_empty(self):
        adapter = self._make_adapter()
        messages = await adapter.parse_webhook({})
        assert messages == []

    # ---- 消息发送路由 ----

    @pytest.mark.asyncio
    async def test_send_message_group(self):
        """群聊消息应该用 openConversationId。"""
        adapter = self._make_adapter()
        adapter._platform.send_message = AsyncMock(
            return_value=_make_im_message(id="sent_001")
        )
        await adapter.send_message(
            "conv_001", "hello group",
            chat_type="group",
        )
        adapter._platform.send_message.assert_called_once_with(
            receive_id="conv_001",
            content="hello group",
            msg_type="text",
            receive_id_type="openConversationId",
        )

    @pytest.mark.asyncio
    async def test_send_message_p2p(self):
        """单聊消息应该用 staffId。"""
        adapter = self._make_adapter()
        adapter._platform.send_message = AsyncMock(
            return_value=_make_im_message(id="sent_002")
        )
        await adapter.send_message(
            "conv_001", "hello p2p",
            sender_id="staff_001",
            chat_type="p2p",
        )
        adapter._platform.send_message.assert_called_once_with(
            receive_id="staff_001",
            content="hello p2p",
            msg_type="text",
            receive_id_type="staffId",
        )


# ===========================================================================
# 企业微信 WecomAdapter
# ===========================================================================

class TestWecomAdapter:

    def _make_adapter_app_mode(self):
        """企业自建应用模式。"""
        from middleware.im.gateway.channels.wecom import WecomAdapter
        adapter = WecomAdapter()
        adapter._config = _make_config(ChannelType.WECOM)
        adapter._platform = MagicMock()
        adapter._receiver = None
        adapter._bot_id = ""
        return adapter

    def _make_adapter_bot_mode(self):
        """智能机器人模式。"""
        from middleware.im.gateway.channels.wecom import WecomAdapter
        adapter = WecomAdapter()
        adapter._config = _make_config(ChannelType.WECOM)
        adapter._platform = None
        adapter._bot_id = "bot_001"
        adapter._bot_secret = "secret_001"
        # mock receiver
        mock_receiver = MagicMock()
        mock_receiver.send_text = AsyncMock()
        mock_receiver._ws = MagicMock()
        mock_receiver._ws.closed = False
        adapter._receiver = mock_receiver
        return adapter

    # ---- 消息格式转换 ----

    def test_convert_to_gateway_message(self):
        adapter = self._make_adapter_app_mode()
        msg_dict = {
            "id": "wecom_001",
            "chat_id": "chat_001",
            "sender_id": "userid_001",
            "content": "企微消息",
            "msg_type": "text",
            "create_time": "1700000000",
            "chat_type": "p2p",
            "event_type": "",
        }
        gw = adapter._convert_to_gateway_message(msg_dict)

        assert gw.chat_id == "chat_001"
        assert gw.sender_id == "userid_001"
        assert gw.content == "企微消息"
        assert gw.channel_type == ChannelType.WECOM
        assert gw.metadata["chat_type"] == "p2p"

    def test_parse_message_text(self):
        adapter = self._make_adapter_app_mode()
        payload = {
            "MsgType": "text",
            "Content": "企微文本",
            "MsgId": "wecom_msg_001",
            "FromUserName": "userid_001",
            "ChatId": "chat_001",
            "CreateTime": "1700000000",
        }
        result = adapter._parse_message(payload)

        assert result["id"] == "wecom_msg_001"
        assert result["chat_id"] == "chat_001"
        assert result["sender_id"] == "userid_001"
        assert result["content"] == "企微文本"

    def test_parse_message_image(self):
        adapter = self._make_adapter_app_mode()
        payload = {
            "MsgType": "image",
            "PicUrl": "https://img.example.com/pic.jpg",
            "FromUserName": "userid_001",
        }
        result = adapter._parse_message(payload)
        assert result["content"] == "https://img.example.com/pic.jpg"

    def test_parse_message_location(self):
        adapter = self._make_adapter_app_mode()
        payload = {
            "MsgType": "location",
            "Label": "北京天安门",
            "FromUserName": "userid_001",
        }
        result = adapter._parse_message(payload)
        assert "北京天安门" in result["content"]

    def test_parse_event_subscribe(self):
        adapter = self._make_adapter_app_mode()
        payload = {
            "MsgType": "event",
            "Event": "subscribe",
            "FromUserName": "userid_001",
        }
        result = adapter._parse_event(payload)
        assert result is not None
        assert result["sender_id"] == "userid_001"
        assert result["event_type"] == "subscribe"
        assert "关注" in result["content"]

    def test_parse_event_click(self):
        adapter = self._make_adapter_app_mode()
        payload = {
            "MsgType": "event",
            "Event": "click",
            "EventKey": "menu_key_1",
            "FromUserName": "userid_001",
        }
        result = adapter._parse_event(payload)
        assert result["event_type"] == "click"
        assert "menu_key_1" in result["content"]

    # ---- Webhook 解析 ----

    @pytest.mark.asyncio
    async def test_parse_webhook_text_message(self):
        adapter = self._make_adapter_app_mode()
        payload = {
            "MsgType": "text",
            "Content": "hello wecom",
            "MsgId": "msg_001",
            "FromUserName": "user_001",
            "ChatId": "chat_001",
            "CreateTime": "1700000000",
        }
        messages = await adapter.parse_webhook(payload)
        assert len(messages) == 1
        assert messages[0].content == "hello wecom"

    @pytest.mark.asyncio
    async def test_parse_webhook_event(self):
        adapter = self._make_adapter_app_mode()
        payload = {
            "MsgType": "event",
            "Event": "enter_chat",
            "FromUserName": "user_001",
            "CreateTime": "1700000000",
        }
        messages = await adapter.parse_webhook(payload)
        assert len(messages) == 1
        assert messages[0].msg_type == "event"

    @pytest.mark.asyncio
    async def test_parse_webhook_xml_string(self):
        """企微有时以 XML 字符串形式回调。"""
        adapter = self._make_adapter_app_mode()
        xml = """<xml>
  <MsgType><![CDATA[text]]></MsgType>
  <Content><![CDATA[xml content]]></Content>
  <FromUserName><![CDATA[user_001]]></FromUserName>
  <MsgId>001</MsgId>
</xml>"""
        messages = await adapter.parse_webhook(xml)
        assert len(messages) == 1
        assert messages[0].content == "xml content"

    @pytest.mark.asyncio
    async def test_parse_webhook_no_msgtype(self):
        adapter = self._make_adapter_app_mode()
        messages = await adapter.parse_webhook({"other": "data"})
        assert messages == []

    # ---- 消息发送路由 ----

    @pytest.mark.asyncio
    async def test_send_message_app_mode_touser(self):
        """应用模式下发送给用户。"""
        adapter = self._make_adapter_app_mode()
        adapter._platform.send_message = AsyncMock(
            return_value=_make_im_message(id="sent_001")
        )
        await adapter.send_message("user_001", "hello user")

        adapter._platform.send_message.assert_called_once_with(
            receive_id="user_001",
            content="hello user",
            msg_type="text",
            receive_id_type="touser",
        )

    @pytest.mark.asyncio
    async def test_send_message_bot_mode_with_reply_func(self):
        """智能机器人模式：优先走 reply_func（单聊必须绑定 req_id）。"""
        adapter = self._make_adapter_bot_mode()

        mock_reply = AsyncMock()
        adapter._reply_funcs["user_001"] = mock_reply

        await adapter.send_message(
            "chat_001", "hello bot",
            sender_id="user_001",
            chat_type="p2p",
        )

        mock_reply.assert_called_once_with("hello bot")
        assert "user_001" not in adapter._reply_funcs  # pop 后已移除

    @pytest.mark.asyncio
    async def test_send_message_bot_mode_group_send_text(self):
        """智能机器人模式：群聊无 reply_func 时用 send_text。"""
        adapter = self._make_adapter_bot_mode()

        await adapter.send_message("group_chat_001", "hello group", chat_type="group")

        adapter._receiver.send_text.assert_called_once_with(
            chat_id="group_chat_001", text="hello group"
        )

    @pytest.mark.asyncio
    async def test_send_message_bot_mode_p2p_no_reply_func(self):
        """智能机器人模式：单聊无 reply_func 时用 send_text(to_user_id)。"""
        adapter = self._make_adapter_bot_mode()

        await adapter.send_message(
            "", "hello p2p user",
            sender_id="user_001",
            chat_type="p2p",
        )

        adapter._receiver.send_text.assert_called_once_with(
            chat_id="", text="hello p2p user", to_user_id="user_001"
        )

    # ---- XML 工具 ----

    def test_xml_to_dict(self):
        import xml.etree.ElementTree as ET
        adapter = self._make_adapter_app_mode()
        xml = "<root><Key>Value</Key><Num>42</Num></root>"
        root = ET.fromstring(xml)
        result = adapter._xml_to_dict(root)
        assert result == {"Key": "Value", "Num": "42"}


# ===========================================================================
# GatewayMessage 协议测试
# ===========================================================================

class TestGatewayMessageProtocol:
    """验证 GatewayMessage 序列化/反序列化的正确性。"""

    def test_to_json_and_from_json(self):
        msg = GatewayMessage(
            type=MessageType.CHANNEL_MESSAGE,
            channel_type=ChannelType.FEISHU,
            channel_account="acc_001",
            chat_id="chat_001",
            sender_id="user_001",
            content="test",
            msg_type="text",
            session_id="sess_001",
        )
        json_str = msg.to_json()
        restored = GatewayMessage.from_json(json_str)

        assert restored.channel_type == ChannelType.FEISHU
        assert restored.chat_id == "chat_001"
        assert restored.sender_id == "user_001"
        assert restored.content == "test"
        assert restored.session_id == "sess_001"

    def test_to_agent_input(self):
        msg = GatewayMessage(
            channel_type=ChannelType.DINGTALK,
            channel_account="acc_dt",
            chat_id="conv_001",
            sender_id="staff_001",
            content="agent test",
            session_id="sess_dt",
        )
        agent_input = msg.to_agent_input()
        assert agent_input["session_id"] == "sess_dt"
        assert agent_input["user_content"] == "agent test"
        assert agent_input["metadata"]["channel_type"] == "dingtalk"
        assert agent_input["metadata"]["chat_id"] == "conv_001"

    def test_create_response(self):
        original = GatewayMessage(
            type=MessageType.CHANNEL_MESSAGE,
            channel_type=ChannelType.WECOM,
            chat_id="chat_wecom",
            session_id="sess_wecom",
            id="orig_id",
        )
        response = original.create_response("agent reply")
        assert response.type == MessageType.AGENT_RESPONSE
        assert response.channel_type == ChannelType.WECOM
        assert response.content == "agent reply"
        assert response.correlation_id == "orig_id"
        assert response.session_id == "sess_wecom"


# ===========================================================================
# 已知 Bug 验证（确认当前行为，为后续修复提供基准）
# ===========================================================================

class TestKnownBugs:
    """
    记录 adapter 层已知 Bug，运行后可确认它们的当前行为。
    修复后这些测试应改为验证正确行为。
    """

    @pytest.mark.asyncio
    async def test_dingtalk_send_to_chat_returns_empty_on_im_message(self):
        """
        Bug: DingTalkAdapter.send_to_chat 对 IMMessage 调用 .get("id")，
        IMMessage 是 dataclass 没有 .get()，会抛 AttributeError。
        """
        from middleware.im.gateway.channels.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter()
        adapter._config = _make_config(ChannelType.DINGTALK)
        adapter._platform = MagicMock()
        adapter._platform.send_message = AsyncMock(
            return_value=_make_im_message(id="dt_sent")
        )

        with pytest.raises(AttributeError):
            await adapter.send_to_chat("conv_001", "hello")

    @pytest.mark.asyncio
    async def test_wecom_send_to_chat_returns_empty_on_im_message(self):
        """
        Bug: WecomAdapter.send_to_chat 同样对 IMMessage 调用 .get("id")
        且 receive_id_type="chat" 应为 "chatid"。
        """
        from middleware.im.gateway.channels.wecom import WecomAdapter
        adapter = WecomAdapter()
        adapter._config = _make_config(ChannelType.WECOM)
        adapter._platform = MagicMock()
        adapter._platform.send_message = AsyncMock(
            return_value=_make_im_message(id="wc_sent")
        )
        adapter._receiver = None

        with pytest.raises(AttributeError):
            await adapter.send_to_chat("chat_001", "hello")

    @pytest.mark.asyncio
    async def test_dingtalk_verify_webhook_no_token_attr(self):
        """
        Bug: DingTalkAdapter.verify_webhook 使用 self.token，
        但 DingTalkAdapter 没有初始化 self.token 属性，会抛 AttributeError。
        """
        from middleware.im.gateway.channels.dingtalk import DingTalkAdapter
        adapter = DingTalkAdapter()
        adapter._config = _make_config(ChannelType.DINGTALK)

        with pytest.raises(AttributeError):
            await adapter.verify_webhook("sig", b"body")

    @pytest.mark.asyncio
    async def test_wecom_verify_webhook_no_token_attr(self):
        """
        Bug: WecomAdapter.verify_webhook 同样使用 self.token，
        但 WecomAdapter 没有初始化 self.token 属性，会抛 AttributeError。
        """
        from middleware.im.gateway.channels.wecom import WecomAdapter
        adapter = WecomAdapter()
        adapter._config = _make_config(ChannelType.WECOM)

        with pytest.raises(AttributeError):
            await adapter.verify_webhook("sig", b"body")

    @pytest.mark.asyncio
    async def test_wecom_do_stop_calls_sync_stop_with_await(self):
        """
        Bug: WecomAdapter._do_stop 用 await self._receiver.stop()，
        但 WecomReceiver.stop() 是同步方法，await 同步方法会触发 TypeError。
        """
        from middleware.im.gateway.channels.wecom import WecomAdapter
        from middleware.im.im_platform.wecom.receiver import WecomReceiver

        adapter = WecomAdapter()
        # 注入一个真实 WecomReceiver 的 stop（同步方法）
        mock_receiver = MagicMock(spec=WecomReceiver)
        mock_receiver.stop = MagicMock(return_value=None)  # 同步方法
        adapter._receiver = mock_receiver
        adapter._running = True

        # await 同步方法不会抛错（返回 None 再 await None 等于 NoneType is not awaitable）
        with pytest.raises(TypeError):
            await adapter._do_stop()
