import logging

COLOR_MAP = {
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
}


def print_colored(text: str, color: str, bold: bool = False) -> None:
    code = COLOR_MAP.get(color, 37)
    if bold:
        code = f"1;{code}"
    print(f"\033[{code}m{text}\033[0m")


def setup_logging(level: int = logging.INFO) -> None:
    """Initialize a simple console logger once.

    Idempotent: if a handler already exists on the root logger, only updates
    its formatter and level.
    """
    root = logging.getLogger()
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        root.addHandler(handler)
    else:
        for h in root.handlers:
            h.setFormatter(fmt)

    root.setLevel(level)


__all__ = ["print_colored", "setup_logging"]
