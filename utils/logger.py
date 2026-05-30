"""
训练日志工具。
提供结构化日志输出（时间戳、模块、级别），同时写文件和控制台。
"""
import logging
import sys
import os
from datetime import datetime


def setup_logger(
    name: str = "chameleon",
    log_dir: str = "logs",
    level: int = logging.INFO,
    to_console: bool = True,
    to_file: bool = True,
) -> logging.Logger:
    """
    创建并配置 logger。

    Args:
        name: logger 名称。
        log_dir: 日志文件目录。
        level: 日志级别。
        to_console: 是否输出到 stderr。
        to_file: 是否写文件（文件名含时间戳）。

    Returns:
        已配置的 logging.Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-5s] [%(module)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if to_console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Log file: %s", log_path)

    return logger


def get_logger(name: str = "chameleon") -> logging.Logger:
    """获取已存在的 logger（不重新配置）。"""
    return logging.getLogger(name)
