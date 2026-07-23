#!/usr/bin/env python3
"""
动态时间序列火灾传播模拟器 - MTT算法

"""

import numpy as np
import rasterio
from rasterio.transform import rowcol
import heapq
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import warnings
import time as time_module
import re

warnings.filterwarnings('ignore')


# ============================================================================
# ============================================================================

class TeeLogger:
    def __init__(self, log_path: str, mode: str = 'w', encoding: str = 'utf-8'):
        self.log_path = log_path
        self.mode = mode
        self.encoding = encoding
        self._original_stdout = None
        self._log_file = None

    def start(self):
        self._original_stdout = sys.stdout
        self._log_file = open(self.log_path, self.mode, encoding=self.encoding)
        sys.stdout = self

    def stop(self):
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def write(self, message):
        if self._original_stdout is not None:
            self._original_stdout.write(message)
        if self._log_file is not None:
            self._log_file.write(message)
            self._log_file.flush()

    def flush(self):
        if self._original_stdout is not None:
            self._original_stdout.flush()
        if self._log_file is not None:
            self._log_file.flush()


# ============================================================================
# ============================================================================

@dataclass
class MTTConfig:
    """
    MTT 模拟全局配置类。

    所有可调参数均在此集中声明，外部调参脚本（executor_tuner.py）
    只需构造不同的 MTTConfig 实例即可，无需修改引擎内部代码。

    参数说明
    --------
    ros_multiplier : float
        ROS 缩放因子，乘在读取的 rate_of_spread 值上。
        作用等同于 CA 模型中的 ros_scale_factor。
        默认 1.0（不缩放）；可与实测过火面积对比后标定。

    step_duration : float
        每个时间步持续时间，单位：分钟。默认 60.0（= 1 小时）。

    num_hours : int
        模拟总小时数。默认 48。

    save_interval : int
        每隔多少小时保存一次中间周界栅格。默认 1。

    max_lb_ratio : float
        椭圆长宽比（LB ratio）上限，clip 的最大值。默认 8.0。
        增大此值允许强风条件下更扁长的椭圆，可能加快极端方向蔓延。

    lb_ros_ref : float
        计算 LB ratio 的参考速度（m/min）。默认 30.0。
        公式：lb = 1 + (ros / lb_ros_ref) * lb_ros_scale

    lb_ros_scale : float
        LB ratio 的斜率系数。默认 3.0。

    ros_filename : str
        子文件夹中 ROS 栅格的文件名。默认 "rate_of_spread.tif"。

    direction_filename : str
        子文件夹中蔓延方向栅格的文件名。默认 "spread_direction.tif"。

    non_burnable_nodata : float
        rate_of_spread 中视为不可燃的 nodata 值。默认 -9999。

    ignition_row : int
        起火点行号（像素坐标）。

    ignition_col : int
        起火点列号（像素坐标）。
    """

    # ---------- 核心调参参数 ----------
    ros_multiplier: float = 1.0          # ROS 缩放因子（新增，对齐 CA.ros_scale_factor）

    # ---------- 椭圆形状参数 ----------
    max_lb_ratio: float = 8.0            # 椭圆长宽比上限（原硬编码）
    lb_ros_ref: float = 30.0             # LB ratio 参考速度 m/min（原硬编码）
    lb_ros_scale: float = 3.0            # LB ratio 斜率（原硬编码）

    # ---------- 时间参数 ----------
    step_duration: float = 60.0          # 每步持续时间（分钟）
    num_hours: int = 48                  # 模拟总小时数

    # ---------- 输出参数 ----------
    save_interval: int = 1               # 中间结果保存间隔（小时）

    # ---------- 文件名 ----------
    ros_filename: str = "rate_of_spread.tif"
    direction_filename: str = "spread_direction.tif"

    # ---------- 起火点 ----------
    ignition_row: int = 0
    ignition_col: int = 0

    # ---------- 数据处理 ----------
    non_burnable_nodata: float = -9999   # nodata 值


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class FireSpreadParams:
    rate_of_spread: np.ndarray
    spread_direction: np.ndarray
    cell_size: float
    nodata: float = -9999


@dataclass
class IgnitionPoint:
    row: int
    col: int
    time: float = 0.0


# ============================================================================
# 椭圆蔓延速度计算器（参数化版本）
# ============================================================================

class EllipticalSpreadCalculatorVectorized:


    def __init__(self, config: MTTConfig):
        self.config = config

    def compute_lb_ratio(self, ros_max: np.ndarray) -> np.ndarray:
        lb = 1.0 + (ros_max / self.config.lb_ros_ref) * self.config.lb_ros_scale
        return np.clip(lb, 1.0, self.config.max_lb_ratio)

    def compute_directional_ros_vectorized(self,
                                           ros_max: np.ndarray,
                                           spread_direction: np.ndarray,
                                           travel_direction: float,
                                           lb_ratio: np.ndarray) -> np.ndarray:
        lb = np.maximum(lb_ratio, 1.0 + 1e-6)
        eccentricity = np.sqrt(1.0 - (1.0 / lb) ** 2)
        angle_diff = travel_direction - spread_direction
        theta = np.radians(angle_diff)
        numerator = 1.0 - eccentricity
        denominator = 1.0 - eccentricity * np.cos(theta)
        ros = ros_max * (numerator / denominator)
        ros = np.where(ros_max > 0, ros, 0.0)
        ros = np.maximum(ros, 0.0)
        return ros




class MTTDijkstraOptimized:

    NEIGHBORS_16 = [
        (-1,  0,   0.00, 1.0000),
        ( 0,  1,  90.00, 1.0000),
        ( 1,  0, 180.00, 1.0000),
        ( 0, -1, 270.00, 1.0000),
        (-1,  1,  45.00, 1.4142),
        ( 1,  1, 135.00, 1.4142),
        ( 1, -1, 225.00, 1.4142),
        (-1, -1, 315.00, 1.4142),
        (-2,  1,  26.57, 2.2361),
        (-1,  2,  63.43, 2.2361),
        ( 1,  2, 116.57, 2.2361),
        ( 2,  1, 153.43, 2.2361),
        ( 2, -1, 206.57, 2.2361),
        ( 1, -2, 243.43, 2.2361),
        (-1, -2, 296.57, 2.2361),
        (-2, -1, 333.43, 2.2361),
    ]
    NEIGHBORS_8 = NEIGHBORS_16

    KNIGHT_INTERMEDIATES = {
        (-2,  1): [(-1,  0), (-1,  1)],
        (-1,  2): [( 0,  1), (-1,  1)],
        ( 1,  2): [( 0,  1), ( 1,  1)],
        ( 2,  1): [( 1,  0), ( 1,  1)],
        ( 2, -1): [( 1,  0), ( 1, -1)],
        ( 1, -2): [( 0, -1), ( 1, -1)],
        (-1, -2): [( 0, -1), (-1, -1)],
        (-2, -1): [(-1,  0), (-1, -1)],
    }

    def __init__(self, params: FireSpreadParams, config: MTTConfig):
        self.params = params
        self.config = config
        self.rows, self.cols = params.rate_of_spread.shape
        self.cell_size = params.cell_size
        self._precompute_speed_fields()

    def _precompute_speed_fields(self):
        print("    预计算方向速度场 (16邻域)...")
        t0 = time_module.time()

        calc = EllipticalSpreadCalculatorVectorized(self.config)
        lb_ratio = calc.compute_lb_ratio(self.params.rate_of_spread)

        self.speed_fields = {}
        self.time_fields = {}

        for dr, dc, direction, dist_factor in self.NEIGHBORS_16:
            speed = calc.compute_directional_ros_vectorized(
                self.params.rate_of_spread,
                self.params.spread_direction,
                direction,
                lb_ratio
            )
            self.speed_fields[direction] = speed

            distance = self.cell_size * dist_factor
            time_field = np.full((self.rows, self.cols), np.inf, dtype=np.float64)
            valid = speed > 0
            time_field[valid] = distance / speed[valid]
            self.time_fields[(dr, dc)] = time_field

        print(f"    速度场预计算完成 (耗时: {time_module.time() - t0:.2f}秒)")

    def compute_arrival_time(self,
                             ignition_points: List[IgnitionPoint],
                             max_time: float = float('inf'),
                             burned_mask: np.ndarray = None
                             ) -> Tuple[np.ndarray, List[IgnitionPoint]]:

        print(f"    开始MTT计算 (Dijkstra v4 - 时空穿梭修复 + 量子隧穿防线)...")
        print(f"    起火点数量: {len(ignition_points)}")
        t0 = time_module.time()

        arrival_time = np.full((self.rows, self.cols), np.inf, dtype=np.float64)
        visited = np.zeros((self.rows, self.cols), dtype=bool)

        if burned_mask is not None:
            visited[burned_mask] = True

        pq = []
        overflow_nodes: Dict[Tuple[int, int], float] = {}

        valid_ignitions = 0
        for igpt in ignition_points:
            if 0 <= igpt.row < self.rows and 0 <= igpt.col < self.cols:
                visited[igpt.row, igpt.col] = False
                arrival_time[igpt.row, igpt.col] = igpt.time
                if igpt.time <= max_time:
                    heapq.heappush(pq, (igpt.time, igpt.row, igpt.col))
                else:
                    key = (igpt.row, igpt.col)
                    if igpt.time < overflow_nodes.get(key, np.inf):
                        overflow_nodes[key] = igpt.time
                valid_ignitions += 1

        print(f"    有效起火点: {valid_ignitions}")
        if valid_ignitions == 0:
            print("    ⚠️ 没有有效的起火点，跳过此时段")
            return arrival_time, []

        while pq:
            current_time, row, col = heapq.heappop(pq)
            if visited[row, col]:
                continue
            visited[row, col] = True

            for dr, dc, direction, _ in self.NEIGHBORS_16:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if visited[nr, nc]:
                    continue

                travel_time = self.time_fields[(dr, dc)][row, col]
                if np.isinf(travel_time):
                    continue
                if self.params.rate_of_spread[nr, nc] <= 0:
                    continue

                intermediates = self.KNIGHT_INTERMEDIATES.get((dr, dc))
                if intermediates is not None:
                    mid_passable = False
                    for mdr, mdc in intermediates:
                        mr, mc = row + mdr, col + mdc
                        if (0 <= mr < self.rows and 0 <= mc < self.cols and
                                self.params.rate_of_spread[mr, mc] > 0):
                            mid_passable = True
                            break
                    if not mid_passable:
                        continue

                new_time = current_time + travel_time
                if new_time < arrival_time[nr, nc]:
                    arrival_time[nr, nc] = new_time
                    if new_time <= max_time:
                        heapq.heappush(pq, (new_time, nr, nc))
                    else:
                        key = (nr, nc)
                        if new_time < overflow_nodes.get(key, np.inf):
                            overflow_nodes[key] = new_time

        reached = np.sum(np.isfinite(arrival_time) & (arrival_time <= max_time))
        print(f"    MTT计算完成: 耗时 {time_module.time() - t0:.2f}秒, 本时段燃烧 {reached:,} 像素")

        next_ignitions = [
            IgnitionPoint(row=r, col=c, time=t - max_time)
            for (r, c), t in overflow_nodes.items()
        ]
        print(f"    接力节点数 (Queue Carry-over): {len(next_ignitions):,} 个")

        return arrival_time, next_ignitions




class TimeSeriesDataLoader:

    def __init__(self, data_dir: str, config: MTTConfig):
        self.data_dir = Path(data_dir)
        self.config = config
        self._scan_timestep_folders()

    def _scan_timestep_folders(self):
        all_entries = sorted(self.data_dir.iterdir())
        timestep_folders = []

        for entry in all_entries:
            if not entry.is_dir():
                continue
            folder_name = entry.name
            if not re.match(r'^\d{8}_\d{4}$', folder_name):
                continue
            ros_path = entry / self.config.ros_filename
            direction_path = entry / self.config.direction_filename
            if not ros_path.exists() or not direction_path.exists():
                continue
            timestep_folders.append({
                'folder': entry,
                'timestamp': folder_name,
                'ros_path': str(ros_path),
                'direction_path': str(direction_path),
            })

        if not timestep_folders:
            raise FileNotFoundError(
                f"在 {self.data_dir} 中未找到有效时间步文件夹"
            )

        self.timesteps = sorted(timestep_folders, key=lambda x: x['timestamp'])
        self.num_timesteps = len(self.timesteps)

        print(f"\n📂 数据加载器初始化完成:")
        print(f"   数据目录: {self.data_dir}")
        print(f"   时间步数: {self.num_timesteps}")
        print(f"   第一个: {self.timesteps[0]['timestamp']}")
        print(f"   最后一个: {self.timesteps[-1]['timestamp']}")

    def get_timestep_data(self, index: int) -> Tuple[str, str, str]:
        if index < 0 or index >= self.num_timesteps:
            raise IndexError(f"时间步索引 {index} 超出范围")
        ts = self.timesteps[index]
        return ts['ros_path'], ts['direction_path'], ts['timestamp']




class DynamicFireSimulator:

    def __init__(self,
                 data_loader: TimeSeriesDataLoader,
                 output_dir: str,
                 config: MTTConfig):
        """
        单位体系说明：
          - ROS 输入: m/min（Rothermel 直接输出，不做×60转换）
          - ros_multiplier 在加载时应用
          - cell_size: m
          - travel_time = distance(m) / ROS(m/min) → 分钟
          - step_duration: 分钟
        """
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_reference_metadata()
        self._initialize_global_state()

    def _load_reference_metadata(self):
        ros_path, _, _ = self.data_loader.get_timestep_data(0)
        with rasterio.open(ros_path) as src:
            self.profile = src.profile.copy()
            self.transform = src.transform
            self.crs = src.crs
            self.rows = src.height
            self.cols = src.width
            self.cell_size = abs(self.transform[0])
            self.nodata = src.nodata if src.nodata is not None else self.config.non_burnable_nodata

        print(f"\n📐 栅格元数据:")
        print(f"   尺寸: {self.rows} x {self.cols}")
        print(f"   分辨率: {self.cell_size} m")
        print(f"   CRS: {self.crs}")

    def _initialize_global_state(self):
        self.global_arrival_time = np.full((self.rows, self.cols), np.inf, dtype=np.float64)
        self.burned_mask = np.zeros((self.rows, self.cols), dtype=bool)
        self.active_front = []
        print(f"   全局状态数组已初始化")

    def _load_timestep_data(self, index: int) -> FireSpreadParams:
        """
        """
        ros_path, direction_path, _ = self.data_loader.get_timestep_data(index)

        with rasterio.open(ros_path) as src:
            rate_of_spread = src.read(1).astype(np.float64)

        with rasterio.open(direction_path) as src:
            spread_direction = src.read(1).astype(np.float64)

        invalid_mask = (
            (rate_of_spread == self.nodata) |
            np.isnan(rate_of_spread) |
            (rate_of_spread < 0)
        )
        rate_of_spread[invalid_mask] = 0

        if self.config.ros_multiplier != 1.0:
            rate_of_spread = rate_of_spread * self.config.ros_multiplier

        return FireSpreadParams(
            rate_of_spread=rate_of_spread,
            spread_direction=spread_direction,
            cell_size=self.cell_size,
            nodata=self.nodata
        )

    def set_initial_ignition_pixel(self, row: int, col: int):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.initial_ignition = [(row, col)]
            print(f"\n🔥 初始起火点: (row={row}, col={col})")
        else:
            raise ValueError(f"起火点坐标超出范围: ({row}, {col})")

    def _save_intermediate_result(self, hour_index: int, timestamp: str):
        perimeter = self.burned_mask.astype(np.uint8)
        perimeter_path = self.output_dir / f"perimeter_hour_{hour_index:03d}_{timestamp}.tif"
        profile = self.profile.copy()
        profile.update({'dtype': 'uint8', 'nodata': 255, 'count': 1})
        profile.pop('predictor', None)
        with rasterio.open(perimeter_path, 'w', **profile) as dst:
            dst.write(perimeter, 1)

    def run_simulation(self) -> Dict:

        num_hours = min(self.config.num_hours, self.data_loader.num_timesteps)

        print(f"\n{'=' * 70}")
        print("🚀 MTT 动态时间序列火灾蔓延模拟 V5.0")
        print(f"{'=' * 70}")
        print(f"  总时间步数       : {num_hours} 小时")
        print(f"  每步持续时间     : {self.config.step_duration} 分钟")
        print(f"  ROS 缩放因子     : ×{self.config.ros_multiplier:.3f}")
        print(f"  椭圆LB上限       : {self.config.max_lb_ratio:.1f}")
        print(f"  LB参考速度       : {self.config.lb_ros_ref:.1f} m/min")
        print(f"  LB斜率           : {self.config.lb_ros_scale:.2f}")
        print(f"  输出目录         : {self.output_dir}")

        sim_start = time_module.time()
        hourly_stats = []
        carry_over_ignitions: Optional[List[IgnitionPoint]] = None

        for hour_idx in range(num_hours):
            hour_start = time_module.time()
            _, _, timestamp = self.data_loader.get_timestep_data(hour_idx)

            print(f"\n{'─' * 50}")
            print(f"⏰ 第 {hour_idx + 1}/{num_hours} 小时 (时间戳: {timestamp})")
            print(f"{'─' * 50}")

            params = self._load_timestep_data(hour_idx)
            ros_valid = params.rate_of_spread[params.rate_of_spread > 0]
            if len(ros_valid) > 0:
                print(f"  ROS(>0): {ros_valid.min():.4f} - {ros_valid.max():.2f} m/min, "
                      f"中位数: {np.median(ros_valid):.2f}")
            else:
                print(f"  ⚠️ 无有效ROS数据")

            if hour_idx == 0:
                ignition_points = [
                    IgnitionPoint(row=r, col=c, time=0.0)
                    for r, c in self.initial_ignition
                ]
                print(f"  初始起火点: {len(ignition_points)} 个")
                for ip in ignition_points:
                    print(f"    ({ip.row},{ip.col}) ROS={params.rate_of_spread[ip.row, ip.col]:.4f} m/min, "
                          f"方向={params.spread_direction[ip.row, ip.col]:.1f}°")
            else:
                ignition_points = carry_over_ignitions
                print(f"  接力节点: {len(ignition_points):,} 个")
                if ignition_points:
                    times = [ip.time for ip in ignition_points]
                    print(f"    时间戳: min={min(times):.2f}, max={max(times):.2f}, "
                          f"mean={np.mean(times):.2f} 分钟")

            if not ignition_points:
                print("  ⚠️ 无活跃火焰前沿，火灾停止蔓延")
                break

            # [V5.0] MTTDijkstraOptimized 接受 config 参数
            mtt = MTTDijkstraOptimized(params, self.config)
            local_arrival_time, next_ignitions = mtt.compute_arrival_time(
                ignition_points=ignition_points,
                max_time=self.config.step_duration,
                burned_mask=self.burned_mask
            )

            carry_over_ignitions = next_ignitions
            print(f"  下一小时接力节点: {len(carry_over_ignitions):,} 个")

            new_burned_total = int(np.sum(
                (local_arrival_time <= self.config.step_duration) & (~self.burned_mask)
            ))
            print(f"  新燃烧像素: {new_burned_total:,}")

            global_time_offset = hour_idx * self.config.step_duration
            newly_burned = (
                np.isfinite(local_arrival_time) &
                (local_arrival_time <= self.config.step_duration) &
                (~self.burned_mask)
            )
            self.global_arrival_time[newly_burned] = (
                global_time_offset + local_arrival_time[newly_burned]
            )
            self.burned_mask[newly_burned] = True

            total_burned = int(np.sum(self.burned_mask))
            burned_area_ha = total_burned * (self.cell_size ** 2) / 10000
            hour_elapsed = time_module.time() - hour_start

            stats = {
                'hour': hour_idx + 1,
                'timestamp': timestamp,
                'new_burned_pixels': new_burned_total,
                'carry_over_nodes': len(carry_over_ignitions),
                'total_burned_pixels': total_burned,
                'burned_area_ha': float(burned_area_ha),
                'elapsed_seconds': float(hour_elapsed)
            }
            hourly_stats.append(stats)

            print(f"  📊 累计: {total_burned:,} 像素 ({burned_area_ha:.2f} 公顷)")
            print(f"  ⏱️ 耗时: {hour_elapsed:.2f} 秒")

            if (hour_idx + 1) % self.config.save_interval == 0:
                self._save_intermediate_result(hour_idx, timestamp)

        total_elapsed = time_module.time() - sim_start
        final_area_ha = float(
            np.sum(self.burned_mask) * (self.cell_size ** 2) / 10000
        )

        print(f"\n{'=' * 70}")
        print("✅ 模拟完成!")
        print(f"{'=' * 70}")
        print(f"  总耗时: {total_elapsed / 60:.2f} 分钟")
        print(f"  最终燃烧面积: {final_area_ha:.2f} 公顷")

        self._save_final_results(hourly_stats, total_elapsed, final_area_ha)

        return {
            'global_arrival_time': self.global_arrival_time,
            'burned_mask': self.burned_mask,
            'burned_area_ha': final_area_ha,
            'hourly_stats': hourly_stats,
            'total_elapsed': total_elapsed
        }

    def _save_final_results(self, hourly_stats: List[Dict],
                             total_elapsed: float, final_area_ha: float):
        print(f"\n💾 保存最终结果...")

        # 全局到达时间
        at_path = self.output_dir / 'global_arrival_time.tif'
        profile_at = self.profile.copy()
        profile_at.update({'dtype': 'float32', 'nodata': -9999, 'count': 1})
        profile_at.pop('predictor', None)
        at_out = self.global_arrival_time.copy()
        at_out[np.isinf(at_out)] = -9999
        with rasterio.open(at_path, 'w', **profile_at) as dst:
            dst.write(at_out.astype('float32'), 1)
        print(f"  ✓ 全局到达时间: {at_path}")

        # 最终燃烧区域
        burned_path = self.output_dir / 'final_burned_area.tif'
        profile_ba = self.profile.copy()
        profile_ba.update({'dtype': 'uint8', 'nodata': 255, 'count': 1})
        profile_ba.pop('predictor', None)
        with rasterio.open(burned_path, 'w', **profile_ba) as dst:
            dst.write(self.burned_mask.astype('uint8'), 1)
        print(f"  ✓ 最终燃烧区域: {burned_path}")

        # 元数据
        metadata = {
            'version': 'MTT-TimeSeries-V5.0',
            'config': {
                'ros_multiplier':    self.config.ros_multiplier,
                'step_duration':     self.config.step_duration,
                'num_hours':         self.config.num_hours,
                'max_lb_ratio':      self.config.max_lb_ratio,
                'lb_ros_ref':        self.config.lb_ros_ref,
                'lb_ros_scale':      self.config.lb_ros_scale,
                'ignition_row':      self.config.ignition_row,
                'ignition_col':      self.config.ignition_col,
            },
            'results': {
                'final_burned_area_ha': final_area_ha,
                'total_elapsed_seconds': total_elapsed,
                'final_burned_pixels': int(np.sum(self.burned_mask)),
            },
            'fixes_applied': [
                'v2: queue_carryover',
                'v3: 16-connectivity + ghost_node_fix',
                'v4: temporal_paradox_fix + quantum_tunneling_fix',
                'v5: ros_unit_fix (m/min direct) + MTTConfig interface standardization',
            ],
            'timestamp': datetime.now().isoformat(),
            'hourly_stats': hourly_stats,
        }
        meta_path = self.output_dir / 'simulation_metadata.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 元数据: {meta_path}")


# ============================================================================
# 便捷入口函数
# ============================================================================

def run_mtt_simulation(data_dir: str,
                       output_dir: str,
                       config: MTTConfig) -> Dict:
    """
    
    """
    data_loader = TimeSeriesDataLoader(data_dir, config)
    simulator = DynamicFireSimulator(data_loader, output_dir, config)
    simulator.set_initial_ignition_pixel(config.ignition_row, config.ignition_col)
    return simulator.run_simulation()


# ============================================================================
# 主函数
# ============================================================================

def main():
    DATA_DIR   = r"E:\Task\fire_simulation\fire_behavior_output_batch_utm"
    OUTPUT_DIR = r"E:\Task\fire_simulation\mtt_timeseries_output"
    LOG_FILE   = os.path.join(OUTPUT_DIR, "simulation_log.txt")

    # ── 所有可调参数集中在 MTTConfig ──────────────────────────────────────
    config = MTTConfig(
        ros_multiplier   = 1.0,    # 调参入口：ROS 缩放因子
        max_lb_ratio     = 8.0,    # 调参入口：椭圆长宽比上限
        lb_ros_ref       = 30.0,   # 调参入口：LB 参考速度 (m/min)
        lb_ros_scale     = 3.0,    # 调参入口：LB 斜率
        step_duration    = 60.0,
        num_hours        = 48,
        save_interval    = 1,
        ignition_row     = 890,
        ignition_col     = 917,
    )
    # ─────────────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = TeeLogger(LOG_FILE)
    logger.start()

    try:
        print(f"📝 日志: {LOG_FILE}")
        print(f"🕐 开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = run_mtt_simulation(DATA_DIR, OUTPUT_DIR, config)

        print("\n" + "=" * 70)
        print("📋 模拟统计摘要")
        print("=" * 70)
        print(f"\n{'小时':>4} | {'时间戳':13} | {'新燃烧像素':>10} | "
              f"{'接力节点':>9} | {'累计面积(ha)':>14} | {'耗时(s)':>8}")
        print("-" * 80)
        for s in results['hourly_stats']:
            print(f"{s['hour']:4d} | {s['timestamp']:13s} | "
                  f"{s['new_burned_pixels']:10,d} | "
                  f"{s['carry_over_nodes']:9,d} | "
                  f"{s['burned_area_ha']:14.2f} | "
                  f"{s['elapsed_seconds']:8.2f}")

        print(f"\n🕐 结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.stop()


if __name__ == "__main__":
    main()