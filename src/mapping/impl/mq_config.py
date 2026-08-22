import os
from typing import NamedTuple

USERNAME_ENV = "HETERO_MQ_USER"
PASSWORD_ENV = "HETERO_MQ_PASS"
HOST_ENV = "HETERO_MQ_HOST"


class MQConfig(NamedTuple):
    username: str
    password: str
    host: str


class MQError(RuntimeError):
    pass


def get_mq_config() -> MQConfig:
    username = os.environ.get(USERNAME_ENV)
    password = os.environ.get(PASSWORD_ENV)
    host = os.environ.get(HOST_ENV)

    if not (username and password and host):
        raise MQError(f"{USERNAME_ENV}, {PASSWORD_ENV}, and {HOST_ENV} must be set as env variables")

    return MQConfig(username=username, password=password, host=host)
