from collections.abc import AsyncGenerator

import redis
from fastapi.exceptions import ResponseValidationError
from redis import Redis
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession

from core.logger import log
from core.settings import conf

master_async_engine = create_async_engine(
    url=conf.master_db_url.unicode_string(),
    future=True,
    echo=conf.debug,
    echo_pool=conf.debug,
    pool_size=conf.database.master.max_size,
    pool_recycle=conf.database.master.pool_recycle,
    pool_timeout=conf.database.master.pool_timeout,
    pool_pre_ping=True,
    max_overflow=conf.database.master.max_overflow,
)

master_async_factory = async_sessionmaker(
    bind=master_async_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=True,
    class_=AsyncSession,
)

async def get_db() -> AsyncGenerator:
    async with master_async_factory() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError | ResponseValidationError:
            raise
        except Exception as e:
            log.error(f"Database-related error: {repr(e)}")

async def get_redis() -> Redis:
    return redis.from_url(
        url=conf.redis_url.unicode_string(),
        encoding="utf-8",
        decode_responses=True,
    )