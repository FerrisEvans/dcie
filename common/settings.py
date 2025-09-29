import pathlib
from typing import List, Tuple, Type

from pydantic import BaseModel, PostgresDsn, RedisDsn, computed_field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource


class DatasourceConfig(BaseModel):
    username: str
    password: str
    host: str
    port: int
    database: str
    min_size: int
    max_size: int
    pool_timeout: int
    pool_recycle: int
    max_overflow: int


class DatabaseConfig(BaseModel):
    master: DatasourceConfig = None
    # TODO multi datasource
    slave: List[DatasourceConfig] = list()


class RedisConfig(BaseModel):
    username: str
    password: str
    host: str
    port: int
    database: str
    ssl: bool = False


class LogConfig(BaseModel):
    rotation: str = "00:00"
    retention: str = "3 days"
    compression: str = "zip"


class HttpConfig(BaseModel):
    enable_cors: bool = True
    allow_origins: List[str] = ["*"]
    allow_headers: List[str] = ["*"]
    allow_methods: List[str] = ["*"]
    allow_credentials: bool = True


class Engine(BaseModel):
    support_file_types: List[str]
    regex_path: str
    vocabulary_path: str
    tmp_base_path: str = f"{pathlib.Path(__file__).resolve().parent.parent.__str__()}/tmp"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file="../config.toml",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )
    debug: bool = True
    api: str = "/api/v1"
    base_dir: str = pathlib.Path(__file__).resolve().parent.parent.__str__()
    log: LogConfig = LogConfig()
    http: HttpConfig = HttpConfig()
    redis: RedisConfig = None
    database: DatabaseConfig = None
    engine: Engine = None

    @computed_field
    @property
    def reg_path(self) -> pathlib.Path:
        return pathlib.Path(f"{self.base_dir}{self.engine.regex_path}")

    @computed_field
    @property
    def sensitive_words_dict_path(self) -> pathlib.Path:
        return pathlib.Path(f"{self.base_dir}{self.engine.vocabulary_path}")

    @computed_field
    @property
    def tmp_upload_dir(self) -> pathlib.Path:
        return pathlib.Path(f"{self.engine.tmp_base_path}/upload")

    @computed_field
    @property
    def redis_url(self) -> RedisDsn:
        return RedisDsn.build(
            scheme="redis",
            username=self.redis.username,
            password=self.redis.password,
            host=self.redis.host,
            port=self.redis.port,
            path=self.redis.database,
        )

    @computed_field
    @property
    def master_db_url(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            # scheme="postgresql+psycopg2",
            username=self.database.master.username,
            password=self.database.master.password,
            host=self.database.master.host,
            port=self.database.master.port,
            path=self.database.master.database,
        )

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls=settings_cls),)
