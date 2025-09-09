from collections.abc import AsyncGenerator

from fastapi.exceptions import ResponseValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.session import AsyncSession

from core.logger import log
from core.settings import conf

master_async_engine = create_async_engine(
    conf.master_db_url,
    future=True,
    echo=conf.debug,
    echo_pool=conf.debug,
    pool_size=conf.pool_size,
    pool_recycle=conf.pool_recycle,
    pool_timeout=conf.pool_timeout,
    pool_pre_ping=True,
    max_overflow=conf.max_overflow,
)

master_async_factory = async_sessionmaker(
    bind=master_async_engine,
    autoflush=False,
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