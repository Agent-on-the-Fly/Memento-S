# Memento-S 自动更新系统

## 功能概述

实现了完整的自动更新系统，支持：

1. **自动检查更新** - 应用启动后自动检查
2. **后台静默下载** - 自动下载更新包到本地缓存
3. **断点续传** - 支持暂停和恢复下载
4. **完整性校验** - 使用 MD5/SHA1/SHA256 校验更新包
5. **用户确认安装** - 下载完成后提示用户确认安装
6. **跨平台支持** - macOS、Windows、Linux

## 配置文件

在 `config.yaml` 中添加 OTA 配置：

```yaml
ota:
  url: "https://your-update-server.com/api/check"
  auto_check: true           # 启动时自动检查
  auto_download: true        # 自动下载更新
  check_interval_hours: 24   # 检查间隔
  notify_on_complete: true   # 下载完成通知
  install_confirmation: true # 安装前确认
```

## OTA 服务器 API

检查更新接口期望的响应格式：

```json
{
  "update_available": true,
  "latest_version": "1.2.0",
  "download_url": "https://example.com/releases/memento-s-v1.2.0.zip",
  "release_notes": "修复了若干问题",
  "published_at": "2024-01-15",
  "size": 52428800,
  "checksum": "abc123..."
}
```

## 支持的更新包格式

### macOS
- `.zip` - ZIP 压缩包（推荐）
- `.tar.gz` - TAR.GZ 压缩包
- `.dmg` - DMG 磁盘镜像（打开手动安装）

### Windows
- `.zip` - ZIP 压缩包（推荐）
- `.exe` - 安装程序（自动运行）
- `.msi` - MSI 安装包（自动运行）

### Linux
- `.zip` - ZIP 压缩包
- `.tar.gz` - TAR.GZ 压缩包
- `.AppImage` - AppImage 包
- `.deb` - Debian 包（需要 sudo）
- `.rpm` - RPM 包（需要 sudo）

## 文件说明

### gui/modules/auto_update_manager.py
核心更新管理器，包含：
- `AutoUpdateManager` - 主管理类
- `UpdateStatus` - 更新状态枚举
- `UpdateInfo` - 更新信息数据类
- `DownloadProgress` - 下载进度追踪
- `UpdateCache` - 缓存元数据管理

主要方法：
- `start_auto_check()` - 启动自动检查
- `check_for_update()` - 手动检查更新
- `download_update()` - 下载更新
- `pause_download()` - 暂停下载
- `resume_download()` - 恢复下载
- `cancel_download()` - 取消下载
- `install_update()` - 安装更新

### gui/modules/update_notifier.py
更新通知 UI，包含：
- `UpdateNotifier` - 通知管理器

主要功能：
- 显示下载进度对话框
- 下载完成通知卡片
- 安装确认对话框
- 安装进度显示

### middleware/config/config_models.py
扩展的 OTA 配置模型：
- `OTAConfig` - 新增自动更新相关配置

## 工作流程

1. **应用启动** → 延迟 10 秒启动更新检查
2. **检查更新** → 向 OTA 服务器发送请求
3. **发现更新** → 自动开始下载（后台静默）
4. **下载进度** → 显示进度条和速度
5. **下载完成** → 显示通知卡片
6. **用户点击** → 显示安装确认对话框
7. **确认安装** → 执行安装脚本并重启应用

## 安装过程

### macOS
1. 解压更新包
2. 查找 .app bundle
3. 生成 shell 脚本进行替换
4. 执行脚本并退出应用

### Windows
1. 解压更新包
2. 查找可执行文件
3. 生成 batch 脚本进行替换
4. 执行脚本并退出应用

### Linux
1. 根据格式解压或运行安装
2. 查找可执行文件
3. 生成 shell 脚本进行替换
4. 执行脚本并退出应用

## 缓存机制

更新包缓存位置：`~/.memento-s/updates/`

缓存文件：
- `cache.json` - 更新元数据
- `update_{version}.{ext}` - 更新包文件

特点：
- 应用重启后可恢复下载
- 避免重复下载相同版本
- 安装完成后自动清理

## 安全特性

1. **版本比较** - 确保只升级不降级
2. **校验和验证** - 验证下载文件完整性
3. **备份机制** - 安装前自动备份当前版本
4. **原子替换** - 使用临时文件确保替换成功

## 错误处理

- 网络错误 - 自动重试机制
- 下载中断 - 支持断点续传
- 校验失败 - 自动删除并重新下载
- 安装失败 - 保留备份可手动恢复

## 调试日志

所有更新操作都有详细的日志记录：
- `[AutoUpdate]` 前缀标识
- 包含状态变更、进度、错误信息
- 查看 `~/.memento-s/logs/` 目录

## 测试命令

```python
# 手动触发更新检查
from gui.modules.auto_update_manager import AutoUpdateManager
manager = AutoUpdateManager()
await manager.check_for_update()

# 查看缓存状态
print(manager.has_cached_update)
print(manager.current_update)

# 清除缓存
manager.clear_cache()
```

## 测试与演示

项目包含完整的测试演示工具：

### 1. 快速演示（无需服务器）
```bash
# 查看完整的更新流程演示
python tests/auto_update/demo.py

# 快速状态演示
python tests/auto_update/demo.py --quick

# 分步骤演示
python tests/auto_update/demo.py --step check
python tests/auto_update/demo.py --step download
python tests/auto_update/demo.py --step install
```

### 2. Mock OTA 服务器
```bash
# 启动测试服务器
python tests/auto_update/mock_ota_server.py

# 访问 http://localhost:8888 查看 API 文档
```

### 3. 集成测试
```bash
# 测试更新检查
python tests/auto_update/test_auto_update.py check

# 测试下载
python tests/auto_update/test_auto_update.py download

# 测试完整流程
python tests/auto_update/test_auto_update.py full

# 测试缓存管理
python tests/auto_update/test_auto_update.py cache

# 交互式演示
python tests/auto_update/test_auto_update.py demo
```

详细说明请查看 `tests/auto_update/README.md`
