import os

TL_THREAD_COUNT_ENV = "TL_THREAD_COUNT"


def get_tl_thread_count() -> int:
    thread_count = os.environ.get("TL_THREAD_COUNT_ENV")

    if not thread_count:
        raise RuntimeError(f"{TL_THREAD_COUNT_ENV} must be set as an env variable")

    try:
        return int(thread_count)
    except ValueError:
        raise RuntimeError(f"{TL_THREAD_COUNT_ENV} must be formatted as a valid integer")
