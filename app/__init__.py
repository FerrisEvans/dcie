from typing import List

from fastapi import APIRouter

from .tools import router as tools_router

routers: List[APIRouter] = [
    tools_router,
]

__all__ = ["routers", ]