# Third-Party SDK Directory

This directory contains vendored third-party SDKs integrated into the opc_memento_s project for local development, debugging, and customization.

## Overview

The `3rd/` directory houses modified or locally maintained versions of external SDKs. These vendored copies provide:

- **Source code visibility** for debugging and understanding internal implementations
- **Hot-reload capability** during development without package reinstallation
- **Custom modifications** tailored to project-specific requirements
- **Version stability** by decoupling from upstream release cycles

## Integration Architecture

### Path Resolution Strategy

SDKs in this directory integrate with the main application through Python's module path resolution system:

```
Application Import Flow:
1. Application code modifies sys.path at runtime
2. 3rd/ directory prepended to Python path
3. Local SDK shadows any system-installed version
4. Imports resolve to local vendored copy
```

### Implementation Pattern

Modules using these SDKs implement the following bootstrap pattern:

```python
import sys
from pathlib import Path

# Calculate path relative to current module
_3RD_DIR = Path(__file__).resolve().parent.parent.parent / "3rd"
if str(_3RD_DIR) not in sys.path:
    sys.path.insert(0, str(_3RD_DIR))

# Import from vendored SDK
from weixin_sdk import WeixinClient
```

This pattern ensures:
- **Portability**: Works regardless of installation location
- **Priority**: Local SDK takes precedence over pip-installed versions
- **Isolation**: Changes do not affect other Python projects

## Access Methods

### Direct Import

Once the path is configured, import SDK components directly:

```python
# Core client
from weixin_sdk import WeixinClient

# Authentication
from weixin_sdk.auth.qr_login import QRLoginManager

# Types and exceptions
from weixin_sdk import (
    WeixinMessage,
    MessageType,
    WeixinSessionExpiredError,
)
```

### Configuration

SDKs in this directory consume the same configuration as their upstream counterparts. Refer to individual SDK documentation for configuration options.

### Fallback Behavior

Application modules implement graceful degradation when vendored SDKs are unavailable:

```python
try:
    from weixin_sdk import WeixinClient
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    # Log warning or use alternative implementation
```

## Development Workflow

### Modifying SDK Code

1. **Edit files** in the appropriate `3rd/<sdk_name>/` subdirectory
2. **Test immediately** - changes take effect without reinstallation
3. **Verify imports** work correctly:
   ```bash
   python -c "from weixin_sdk import WeixinClient; print('Import successful')"
   ```

### Testing Changes

Run the application components that consume the SDK:

```bash
# Test gateway adapter
python -m middleware.im.gateway.channels.wechat_ilinkai

# Test CLI commands
memento wechat status
```

### Syncing with Upstream

When updating from the original SDK source:

1. **Backup local modifications**:
   ```bash
   cp -r 3rd/weixin_sdk 3rd/weixin_sdk.backup
   ```

2. **Copy updated source**:
   ```bash
   python -c "
   import shutil
   shutil.rmtree('3rd/weixin_sdk')
   shutil.copytree('/path/to/original/weixin_sdk', '3rd/weixin_sdk')
   "
   ```

3. **Re-apply local modifications** from backup

4. **Verify functionality** with application test suite

## SDK Details

### weixin_sdk

**Source**: Local copy of `openclaw-weixin-python`

**Official Repository**: https://github.com/yzailab/openclaw-weixin-python

**Purpose**: WeChat integration via iLink AI API

**Key Components**:

| Module | Description |
|--------|-------------|
| `client.py` | `WeixinClient` - primary API client for HTTP communication |
| `auth/` | QR code login and authentication flows |
| `messaging/` | Message sending, receiving, and command handling |
| `media/` | Media upload, download, and transcoding |
| `storage/` | Session persistence and sync buffer management |
| `types.py` | Data models and type definitions |
| `exceptions.py` | SDK-specific exception classes |
| `webhook.py` | Webhook server for receiving events |
| `retry.py` | Retry logic with circuit breaker pattern |
| `rate_limiter.py` | Token bucket and sliding window rate limiting |

**Usage in Project**:

- `middleware/im/gateway/channels/wechat_ilinkai.py` - Gateway adapter for WeChat channel
- `cli/commands/wechat.py` - CLI commands for WeChat management
- `gui/widgets/wechat_login_dialog.py` - GUI login dialog
- `gui/modules/settings_panel.py` - Settings configuration panel

**API Surface**:

```python
# Primary exports from weixin_sdk
WeixinClient                    # Main API client
WeixinAccount / WeixinAccountManager  # Account management
MediaUploader / MediaSender     # Media handling
WebhookServer / WebhookHandler  # Event handling
RetryConfig / CircuitBreaker    # Resilience patterns
RateLimiter / TokenBucket       # Rate limiting
```

## Best Practices

### Do

- **Document modifications** with inline comments explaining why changes were made
- **Track local changes** using version control to enable diffing against upstream
- **Maintain compatibility** with the upstream API to ease future updates
- **Test thoroughly** after any SDK modification before committing
- **Use feature branches** when making significant SDK modifications

### Do Not

- **Never commit generated files** (__pycache__, .pyc) to version control
- **Avoid breaking API changes** without updating all consuming code
- **Do not modify vendored code** for features that could be implemented in the application layer
- **Never expose sensitive data** (tokens, keys) in SDK modifications

### Version Management

Maintain a `CHANGES.md` or comments in modified files documenting:
- What was changed from upstream
- Why the change was necessary
- Date of modification
- Related issue or feature request

### Security Considerations

Vendored SDKs receive the same security scrutiny as application code:
- Review all modifications for security implications
- Keep SDKs updated with upstream security patches
- Audit vendored code before initial integration
