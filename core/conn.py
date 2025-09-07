from redis.asyncio import Redis
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, AsyncAttrs

from config.settings import MASTER_DB_URL, REDIS_URL


