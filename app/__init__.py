from typing import List

from fastapi import APIRouter

from .uploader import router as upload_router

routers: List[APIRouter] = [
    upload_router,
]

__all__ = ["routers", ]