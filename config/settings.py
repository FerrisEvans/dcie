from enum import Enum
from urllib .parse import quote_plus

class Env(Enum):
    DEV = {"file": "config/profile/dev.py"}
    PROD = {"file": "config/profile/prod.py"}

env = Env.DEV

match env:
    case Env.PROD:
        from config.profile.prod import *
    case _:
        from config.profile.dev import *


MASTER_DB_URL = f"postgresql+psycopg2://{PG_USER}:{quote_plus(PG_PWD)}@{PG_HOST}:{PG_PORT}/{PG_DB}"
# TODO multi datasource
SLAVE1_DB_URL = ""
SLAVE2_DB_URL = ""
REDIS_URL = f"redis://{REDIS_USER}@{quote_plus(REDIS_PWD)}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
# 日志配置
LOG_ROTATION = "00:00"
LOG_RETENTION = "3 days"
LOG_COMPRESSION = "zip"
# 是否启用跨域
CORS_ORIGIN_ENABLE = True
# 允许访问的域名列表
ALLOW_ORIGINS = ["*"]
# 是否支持携带 cookie
ALLOW_CREDENTIALS = True
# 允许跨域的 http 方法， get / post / put 等
ALLOW_METHODS = ["*"]
# 允许迭代的headers，可以用来鉴别来源等作用
ALLOW_HEADERS = ["*"]
