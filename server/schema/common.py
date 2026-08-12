# SPDX-License-Identifier: Apache-2.0
"""
server/schema/common.py
通用 Pydantic 模型
"""
from pydantic import BaseModel
from typing import Any, Optional


class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = True


class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = False
    error: str
    detail: Optional[str] = None
