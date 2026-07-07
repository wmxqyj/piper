"""
手眼标定模块

功能：
- 棋盘格角点检测与位姿估计 (board_detector)
- 手眼标定数据采集 (data_collector)
- AX=XB 求解器 (solver)
- 标定结果管理 (result_manager)
"""

from .board_detector import ChessboardDetector
from .data_collector import HandEyeDataCollector
from .solver import HandEyeSolver
from .result_manager import CalibrationResult

__all__ = [
    "ChessboardDetector",
    "HandEyeDataCollector",
    "HandEyeSolver",
    "CalibrationResult",
]
