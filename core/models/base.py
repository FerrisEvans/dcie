import datetime

from asyncpg import UniqueViolationError
from fastapi import HTTPException, status

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, declared_attr, Mapped, mapped_column


class BaseModel(DeclarativeBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement="auto")
    delete: Mapped[bool] = mapped_column(default=False)
    created_time: Mapped[datetime.datetime] = mapped_column(default=lambda: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    @declared_attr.directive
    def __tablename__(self) -> str:
        table_name = self.__tablename__.__str__()
        if not table_name:
            model_name = self.__class__.__name__
            ls = []
            for index, char in enumerate(model_name):
                if char.isupper() and index != 0:
                    ls.append("_")
                ls.append(char)
            table_name = "".join(ls).lower()
        return table_name

    async def save(self, db_session: AsyncSession):
        db_session.add(self)
        await db_session.flush()
        await db_session.refresh(self)
        return self

    # async def delete(self, db_session: AsyncSession):
    #     try:
    #         await db_session.delete(self)
    #         return True
    #     except SQLAlchemyError as ex:
    #         raise HTTPException(
    #             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=repr(ex)
    #         ) from ex
    #
    # async def update(self, **kwargs):
    #     try:
    #         for k, v in kwargs.items():
    #             setattr(self, k, v)
    #         return True
    #     except SQLAlchemyError as ex:
    #         raise HTTPException(
    #             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=repr(ex)
    #         ) from ex
    #
    # async def save_or_update(self, db_session: AsyncSession):
    #     try:
    #         db_session.add(self)
    #         await db_session.flush()
    #         return True
    #     except IntegrityError as exception:
    #         if isinstance(exception.orig, UniqueViolationError):
    #             return await db_session.merge(self)
    #         else:
    #             raise HTTPException(
    #                 status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    #                 detail=repr(exception),
    #             ) from exception