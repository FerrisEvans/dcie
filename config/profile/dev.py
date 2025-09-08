from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEBUG = True

PG_USER = "postgres"
PG_PWD = "postgres"
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "dcie"
REDIS_USER = "admin"
REDIS_PWD = "admin"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_SSL = True

SUBSCRIBE = "dcie_dev_queue"