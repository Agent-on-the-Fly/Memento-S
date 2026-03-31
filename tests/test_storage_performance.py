#!/usr/bin/env python3
"""
Storage Service 性能测试

测试 SessionService 和 MessageService 的性能表现

使用方法:
    .venv/bin/python tests/test_storage_performance.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from middleware.storage import (
    Base,
    MessageService,
    SessionService,
    SessionCreate,
    MessageCreate,
    SessionUpdate,
)
from middleware.storage.core.engine import get_db_manager
from utils.logger import setup_logger


class PerformanceMetrics:
    """性能指标收集器"""

    def __init__(self, name: str):
        self.name = name
        self.times: list[float] = []

    def add(self, duration: float):
        """添加一次执行时间"""
        self.times.append(duration)

    def summary(self) -> dict:
        """返回性能摘要"""
        if not self.times:
            return {"name": self.name, "count": 0}

        return {
            "name": self.name,
            "count": len(self.times),
            "total_time": sum(self.times),
            "avg_time": mean(self.times),
            "median_time": median(self.times),
            "min_time": min(self.times),
            "max_time": max(self.times),
            "std_dev": stdev(self.times) if len(self.times) > 1 else 0,
            "ops_per_sec": len(self.times) / sum(self.times)
            if sum(self.times) > 0
            else 0,
        }

    def print_summary(self):
        """打印性能摘要"""
        s = self.summary()
        print(f"\n【{self.name}】")
        print(f"  执行次数: {s['count']}")
        print(f"  总耗时: {s['total_time']:.3f}s")
        print(f"  平均耗时: {s['avg_time'] * 1000:.2f}ms")
        print(f"  中位数: {s['median_time'] * 1000:.2f}ms")
        print(f"  最小值: {s['min_time'] * 1000:.2f}ms")
        print(f"  最大值: {s['max_time'] * 1000:.2f}ms")
        print(f"  标准差: {s['std_dev'] * 1000:.2f}ms")
        print(f"  每秒操作: {s['ops_per_sec']:.1f} ops/sec")


async def test_session_service_performance():
    """测试 SessionService 性能"""
    print("\n" + "=" * 70)
    print("SessionService 性能测试")
    print("=" * 70)

    session_service = SessionService()

    # 测试 1: 创建会话性能
    create_metrics = PerformanceMetrics("创建会话 (create)")
    session_ids = []

    test_count = 50
    print(f"\n测试创建 {test_count} 个会话...")

    for i in range(test_count):
        start = time.perf_counter()
        session = await session_service.create(
            SessionCreate(title=f"性能测试会话 {i}", description=f"测试描述 {i}")
        )
        elapsed = time.perf_counter() - start
        create_metrics.add(elapsed)
        session_ids.append(session.id)

    create_metrics.print_summary()

    # 测试 2: 获取会话性能
    get_metrics = PerformanceMetrics("获取会话 (get)")

    print(f"\n测试获取 {test_count} 个会话...")
    for session_id in session_ids:
        start = time.perf_counter()
        session = await session_service.get(session_id)
        elapsed = time.perf_counter() - start
        get_metrics.add(elapsed)

    get_metrics.print_summary()

    # 测试 3: 更新会话性能
    update_metrics = PerformanceMetrics("更新会话 (update)")

    print(f"\n测试更新 {test_count} 个会话...")
    for session_id in session_ids:
        start = time.perf_counter()
        updated = await session_service.update(
            session_id, SessionUpdate(title="更新后的标题")
        )
        elapsed = time.perf_counter() - start
        update_metrics.add(elapsed)

    update_metrics.print_summary()

    # 测试 4: 列出最近会话性能
    list_metrics = PerformanceMetrics("列出最近会话 (list_recent)")

    print(f"\n测试列出最近会话 (重复 {test_count} 次)...")
    for _ in range(test_count):
        start = time.perf_counter()
        sessions = await session_service.list_recent(limit=20)
        elapsed = time.perf_counter() - start
        list_metrics.add(elapsed)

    list_metrics.print_summary()

    # 测试 5: 批量创建性能测试
    print(f"\n测试批量创建 100 个会话...")
    batch_start = time.perf_counter()
    batch_ids = []

    for i in range(100):
        session = await session_service.create(SessionCreate(title=f"批量测试会话 {i}"))
        batch_ids.append(session.id)

    batch_elapsed = time.perf_counter() - batch_start
    print(f"  总耗时: {batch_elapsed:.3f}s")
    print(f"  平均每个: {batch_elapsed / 100 * 1000:.2f}ms")
    print(f"  每秒创建: {100 / batch_elapsed:.1f} 个")

    # 清理：删除测试数据
    print(f"\n清理测试数据...")
    cleanup_start = time.perf_counter()
    all_ids = session_ids + batch_ids
    for sid in all_ids:
        await session_service.delete(sid)
    cleanup_elapsed = time.perf_counter() - cleanup_start
    print(f"  删除 {len(all_ids)} 个会话耗时: {cleanup_elapsed:.3f}s")

    return {
        "create": create_metrics.summary(),
        "get": get_metrics.summary(),
        "update": update_metrics.summary(),
        "list_recent": list_metrics.summary(),
        "batch_create_100": {
            "total_time": batch_elapsed,
            "avg_time": batch_elapsed / 100,
            "ops_per_sec": 100 / batch_elapsed,
        },
    }


async def test_message_service_performance():
    """测试 MessageService 性能"""
    print("\n" + "=" * 70)
    print("MessageService 性能测试")
    print("=" * 70)

    session_service = SessionService()
    message_service = MessageService()

    # 创建测试会话
    print("\n创建测试会话...")
    session = await session_service.create(SessionCreate(title="消息性能测试会话"))
    session_id = session.id
    print(f"  会话 ID: {session_id}")

    # 测试 1: 创建消息性能
    create_metrics = PerformanceMetrics("创建消息 (create)")
    message_ids = []

    test_count = 100
    print(f"\n测试创建 {test_count} 条消息...")

    for i in range(test_count):
        start = time.perf_counter()
        msg = await message_service.create(
            MessageCreate(
                session_id=session_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"这是测试消息内容 {i}" * 10,  # 较长内容
                meta_info={"tokens": 50},
            )
        )
        elapsed = time.perf_counter() - start
        create_metrics.add(elapsed)
        message_ids.append(msg.id)

    create_metrics.print_summary()

    # 测试 2: 获取消息性能
    get_metrics = PerformanceMetrics("获取消息 (get)")

    print(f"\n测试获取 {test_count} 条消息...")
    for msg_id in message_ids:
        start = time.perf_counter()
        msg = await message_service.get(msg_id)
        elapsed = time.perf_counter() - start
        get_metrics.add(elapsed)

    get_metrics.print_summary()

    # 测试 3: 列出会话消息性能
    list_metrics = PerformanceMetrics("列出会话消息 (list_by_session)")

    print(f"\n测试列出会话消息 (重复 50 次)...")
    for _ in range(50):
        start = time.perf_counter()
        messages = await message_service.list_by_session(session_id)
        elapsed = time.perf_counter() - start
        list_metrics.add(elapsed)

    list_metrics.print_summary()

    # 测试 4: 大消息内容测试
    print(f"\n测试大消息内容创建...")
    large_content = "这是一个大消息内容。" * 1000  # 约 11KB
    large_metrics = PerformanceMetrics("创建大消息 (11KB)")

    for i in range(20):
        start = time.perf_counter()
        msg = await message_service.create(
            MessageCreate(
                session_id=session_id,
                role="user",
                content=large_content,
                meta_info={"tokens": 3000},
            )
        )
        elapsed = time.perf_counter() - start
        large_metrics.add(elapsed)

    large_metrics.print_summary()

    # 测试 5: 批量消息创建性能
    print(f"\n测试批量创建 200 条消息...")
    batch_start = time.perf_counter()
    batch_ids = []

    for i in range(200):
        msg = await message_service.create(
            MessageCreate(
                session_id=session_id,
                role="assistant",
                content=f"批量消息 {i}",
            )
        )
        batch_ids.append(msg.id)

    batch_elapsed = time.perf_counter() - batch_start
    print(f"  总耗时: {batch_elapsed:.3f}s")
    print(f"  平均每条: {batch_elapsed / 200 * 1000:.2f}ms")
    print(f"  每秒创建: {200 / batch_elapsed:.1f} 条")

    # 再次测试列出消息性能（有大量消息后）
    print(f"\n测试大量消息后列出性能...")
    large_list_metrics = PerformanceMetrics("列出大量消息 (320条)")

    for _ in range(20):
        start = time.perf_counter()
        messages = await message_service.list_by_session(session_id)
        elapsed = time.perf_counter() - start
        large_list_metrics.add(elapsed)

    large_list_metrics.print_summary()
    print(f"  实际获取消息数: {len(messages)} 条")

    # 清理
    print(f"\n清理测试数据...")
    cleanup_start = time.perf_counter()
    await session_service.delete(session_id)
    cleanup_elapsed = time.perf_counter() - cleanup_start
    print(f"  删除会话及关联消息耗时: {cleanup_elapsed:.3f}s")

    return {
        "create": create_metrics.summary(),
        "get": get_metrics.summary(),
        "list_by_session": list_metrics.summary(),
        "create_large": large_metrics.summary(),
        "batch_create_200": {
            "total_time": batch_elapsed,
            "avg_time": batch_elapsed / 200,
            "ops_per_sec": 200 / batch_elapsed,
        },
        "list_large": large_list_metrics.summary(),
    }


async def test_concurrent_performance():
    """测试并发性能"""
    print("\n" + "=" * 70)
    print("并发性能测试")
    print("=" * 70)

    session_service = SessionService()
    message_service = MessageService()

    # 创建测试会话
    session = await session_service.create(SessionCreate(title="并发测试会话"))
    session_id = session.id

    # 测试 1: 并发创建消息
    print("\n测试并发创建消息 (10 个并发，每个 20 条)...")

    async def create_batch(batch_id: int):
        """创建一批消息"""
        ids = []
        for i in range(20):
            msg = await message_service.create(
                MessageCreate(
                    session_id=session_id,
                    role="user",
                    content=f"并发测试消息 batch={batch_id} msg={i}",
                )
            )
            ids.append(msg.id)
        return ids

    start = time.perf_counter()
    tasks = [create_batch(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    total_messages = sum(len(r) for r in results)
    print(f"  总耗时: {elapsed:.3f}s")
    print(f"  创建消息数: {total_messages} 条")
    print(f"  每秒创建: {total_messages / elapsed:.1f} 条")
    print(f"  平均每条: {elapsed / total_messages * 1000:.2f}ms")

    # 测试 2: 并发获取消息
    print("\n测试并发获取消息...")
    all_message_ids = [msg_id for batch in results for msg_id in batch]

    async def get_message(msg_id: str):
        """获取单条消息"""
        return await message_service.get(msg_id)

    start = time.perf_counter()
    tasks = [get_message(msg_id) for msg_id in all_message_ids]
    messages = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    print(f"  总耗时: {elapsed:.3f}s")
    print(f"  获取消息数: {len(messages)} 条")
    print(f"  每秒获取: {len(messages) / elapsed:.1f} 条")
    print(f"  平均每条: {elapsed / len(messages) * 1000:.2f}ms")

    # 清理
    await session_service.delete(session_id)

    return {
        "concurrent_create": {
            "total_time": elapsed,
            "total_messages": total_messages,
            "ops_per_sec": total_messages / elapsed,
        },
        "concurrent_get": {
            "total_time": elapsed,
            "total_messages": len(messages),
            "ops_per_sec": len(messages) / elapsed,
        },
    }


async def main():
    """主测试函数"""
    print("=" * 70)
    print("Storage Service 性能测试")
    print("=" * 70)

    # 初始化日志
    setup_logger()

    # 初始化数据库
    from middleware.config.config_manager import ConfigManager

    manager = ConfigManager()
    db_path = manager.get_db_path()
    db_url = f"sqlite+aiosqlite:///{db_path}"

    print(f"\n数据库路径: {db_path}")

    db_manager = get_db_manager()
    await db_manager.init(db_url=db_url, echo=False)

    # 创建表
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ 数据库初始化完成")

    # 运行性能测试
    results = {}

    try:
        # Session 服务测试
        results["session"] = await test_session_service_performance()

        # Message 服务测试
        results["message"] = await test_message_service_performance()

        # 并发测试
        results["concurrent"] = await test_concurrent_performance()

    finally:
        # 打印汇总报告
        print_summary_report(results)


def print_summary_report(results: dict):
    """打印汇总报告"""
    print("\n" + "=" * 70)
    print("性能测试汇总报告")
    print("=" * 70)

    # Session Service 汇总
    if "session" in results:
        print("\n【SessionService】")
        session = results["session"]

        if "create" in session:
            s = session["create"]
            print(
                f"  创建会话: {s['avg_time'] * 1000:.2f}ms/次 ({s['ops_per_sec']:.1f} ops/sec)"
            )

        if "get" in session:
            s = session["get"]
            print(
                f"  获取会话: {s['avg_time'] * 1000:.2f}ms/次 ({s['ops_per_sec']:.1f} ops/sec)"
            )

        if "update" in session:
            s = session["update"]
            print(
                f"  更新会话: {s['avg_time'] * 1000:.2f}ms/次 ({s['ops_per_sec']:.1f} ops/sec)"
            )

        if "list_recent" in session:
            s = session["list_recent"]
            print(
                f"  列出会话: {s['avg_time'] * 1000:.2f}ms/次 ({s['ops_per_sec']:.1f} ops/sec)"
            )

        if "batch_create_100" in session:
            s = session["batch_create_100"]
            print(
                f"  批量创建: {s['avg_time'] * 1000:.2f}ms/次 ({s['ops_per_sec']:.1f} ops/sec)"
            )

    # Message Service 汇总
    if "message" in results:
        print("\n【MessageService】")
        msg = results["message"]

        if "create" in msg:
            s = msg["create"]
            print(
                f"  创建消息: {s['avg_time'] * 1000:.2f}ms/次 ({s['ops_per_sec']:.1f} ops/sec)"
            )

        if "get" in msg:
            s = msg["get"]
            print(
                f"  获取消息: {s['avg_time'] * 1000:.2f}ms/次 ({s['ops_per_sec']:.1f} ops/sec)"
            )

        if "list_by_session" in msg:
            s = msg["list_by_session"]
            print(
                f"  列出消息: {s['avg_time'] * 1000:.2f}ms/次 ({s['ops_per_sec']:.1f} ops/sec)"
            )

        if "create_large" in msg:
            s = msg["create_large"]
            print(
                f"  创建大消息: {s['avg_time'] * 1000:.2f}ms/次 ({s['ops_per_sec']:.1f} ops/sec)"
            )

        if "list_large" in msg:
            s = msg["list_large"]
            print(
                f"  列出大量消息: {s['avg_time'] * 1000:.2f}ms/次 ({s['ops_per_sec']:.1f} ops/sec)"
            )

    # 并发测试汇总
    if "concurrent" in results:
        print("\n【并发测试】")
        conc = results["concurrent"]

        if "concurrent_create" in conc:
            s = conc["concurrent_create"]
            print(f"  并发创建: {s['ops_per_sec']:.1f} ops/sec")

        if "concurrent_get" in conc:
            s = conc["concurrent_get"]
            print(f"  并发获取: {s['ops_per_sec']:.1f} ops/sec")

    print("\n" + "=" * 70)
    print("性能测试完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
