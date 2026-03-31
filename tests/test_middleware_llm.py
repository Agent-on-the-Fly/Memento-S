#!/usr/bin/env python3
"""
测试 middleware/llm 模块

使用方法:
    # 使用默认 profile (default)
    .venv/bin/python tests/test_middleware_llm.py

    # 使用指定 profile
    .venv/bin/python tests/test_middleware_llm.py --profile kimi-vllm

    # 仅测试初始化
    .venv/bin/python tests/test_middleware_llm.py --init-only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from middleware.config.config_manager import ConfigManager
from middleware.llm import LLMClient, LLMResponse, LLMStreamChunk
from utils.logger import setup_logger, logger


async def test_llm_client(profile: str | None = None, init_only: bool = False):
    """测试 LLM 客户端

    Args:
        profile: 指定使用的 LLM profile，None 则使用默认
        init_only: 仅测试初始化，不进行实际调用
    """
    print("=" * 70)
    print("测试 middleware/llm 模块")
    print("=" * 70)

    # 初始化日志
    setup_logger()

    # 准备配置管理器
    manager = ConfigManager()

    # 先加载配置检查可用 profiles
    config = manager.load()
    available_profiles = list(config.llm.profiles.keys())

    if profile:
        print(f"\n使用 Profile: {profile}")
        if profile in available_profiles:
            # 通过 set 方法修改 active_profile 并保存
            manager.set("llm.active_profile", profile, save=True)
            print(f"  (原 profile: {config.llm.active_profile})")
            # 重新加载以获取更新后的配置
            config = manager.load()
        else:
            print(f"  ✗ Profile '{profile}' 不存在")
            print(f"  可用 profiles: {available_profiles}")
            profile = None
    else:
        print(f"\n使用默认 Profile: {config.llm.active_profile}")
        print(f"  可用 profiles: {available_profiles}")

    try:
        print("\n【1. 初始化 LLM 客户端】")
        # 传入已加载配置的 manager
        client = LLMClient(config_manager=manager)
        print(f"✓ 客户端初始化成功")
        print(f"  Model: {client.model}")
        print(f"  Base URL: {client.base_url or '默认'}")
        print(f"  Timeout: {client.timeout}s")
        print(f"  Max tokens: {client.max_tokens}")
        print(f"  Temperature: {client.temperature}")
        print(f"  API Key: {'已设置' if client.api_key else '未设置'}")
        print(f"  Extra Headers: {client.extra_headers}")
        print(f"  Extra Body: {client.extra_body}")

        # 打印调用参数示例
        print("\n【调用参数示例】")
        sample_kwargs = client._build_completion_kwargs(
            messages=[{"role": "user", "content": "Hello"}], stream=False
        )
        import json

        # 隐藏敏感信息
        safe_kwargs = dict(sample_kwargs)
        if "api_key" in safe_kwargs:
            safe_kwargs["api_key"] = "***" if safe_kwargs["api_key"] else None
        print(f"  {json.dumps(safe_kwargs, indent=4, ensure_ascii=False)}")

        if init_only:
            print("\n【初始化完成，跳过 API 调用】")
            return

        print("\n【2. 测试非流式调用】")
        print("  发送: 'Hello, how are you?'")

        try:
            response = await client.async_chat(
                messages=[{"role": "user", "content": "Hello, how are you?"}],
            )
            print(f"✓ 响应成功")
            print(f"  Content: {response.text[:100]}...")
            print(f"  Has tool calls: {response.has_tool_calls}")
            print(f"  Finish reason: {response.finish_reason}")
        except Exception as e:
            print(f"✗ 调用失败: {e}")

        print("\n【3. 测试流式调用】")
        print("  发送: 'Tell me a short story'")

        try:
            print("  流式响应: ", end="", flush=True)
            content_parts = []

            async for chunk in client.async_stream_chat(
                messages=[
                    {"role": "user", "content": "Tell me a short story in one sentence"}
                ],
            ):
                if chunk.delta_content:
                    print(chunk.delta_content, end="", flush=True)
                    content_parts.append(chunk.delta_content)

                if chunk.is_finished:
                    print(f"\n  ✓ 流结束，原因: {chunk.finish_reason}")

            full_content = "".join(content_parts)
            print(f"\n  完整内容长度: {len(full_content)} chars")
        except Exception as e:
            print(f"\n✗ 流式调用失败: {e}")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="测试 middleware/llm 模块")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="指定 LLM profile (如: default, kimi-vllm, fast)",
    )
    parser.add_argument(
        "--init-only", action="store_true", help="仅测试初始化，不进行实际 API 调用"
    )

    args = parser.parse_args()

    asyncio.run(test_llm_client(profile=args.profile, init_only=args.init_only))


if __name__ == "__main__":
    main()
