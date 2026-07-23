#!/usr/bin/env python3
"""
CA（元胞自动机）森林火灾蔓延模型
=============================================================================


"""

import numpy as np
import rasterio
import rasterio.warp
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import json
import csv
import warnings
import time as time_module
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

try:
    from numba import njit
    NUMBA_AVAILABLE = True
    print("✓ Numba 已检测到，核心传播函数将使用 JIT 编译加速")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠ 未检测到 Numba，使用纯 Python 模式（建议: pip install numba）")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator

UNBURNED     = np.int8(0)
BURNING      = np.int8(1)
BURNED_OUT   = np.int8(2)
NON_BURNABLE = np.int8(3)


@dataclass
class CAConfig:
    dt_minutes: float = 1.0
    burn_duration_minutes: float = 60.0
    pixel_size_m: float = None
    ros_unit: str = 'm_per_min'

    # ------------------------------------------------------------------ #
    # [Fix D] 椭圆偏心率                                        #
        # ------------------------------------------------------------------ #
    eccentricity: Optional[float] = None          # 新增 Fix D

    back_to_head_ratio: float = 0.2
    direction_weight_power: float = 2.0
    min_ros_threshold: float = 0.01

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    ros_scale_factor: float = 1.0                 # 新增 Fix E

    base_ignition_prob: float = 1.0
    use_fractional_accumulation: bool = True
    use_meteo_interpolation: bool = True
    use_adaptive_timestep: bool = True
    max_cfl: float = 0.9
    ignition_points: List[Tuple[int, int]] = field(default_factory=list)
    ignition_geojson: Optional[str] = None
    save_state_every_n_steps: int = 60
    save_arrival_time: bool = True
    save_intensity_map: bool = True
    verbose: bool = True
    non_burnable_codes: List[int] = field(
        default_factory=lambda: [91, 92, 93, 98, 99]
    )


_NB_OFFSETS_R = np.array([-1, -1,  0,  1,  1,  1,  0, -1], dtype=np.int32)
_NB_OFFSETS_C = np.array([ 0,  1,  1,  1,  0, -1, -1, -1], dtype=np.int32)
_NB_ANGLES    = np.array([0., 45., 90., 135., 180., 225., 270., 315.], dtype=np.float64)
_NB_DIST_FAC  = np.array([1., 1.4142, 1., 1.4142, 1., 1.4142, 1., 1.4142], dtype=np.float64)

NEIGHBOR_OFFSETS = list(zip(_NB_OFFSETS_R.tolist(), _NB_OFFSETS_C.tolist()))


@njit(cache=True)
def _compute_ellipse_ros_multipliers(spread_dir: float, eccentricity: float) -> np.ndarray:
    multipliers = np.empty(8, dtype=np.float64)
    e = eccentricity
    for k in range(8):
        diff = (_NB_ANGLES[k] - spread_dir) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        cos_diff = np.cos(np.radians(diff))
        denom = 1.0 - e * cos_diff
        if denom < 1e-8:
            denom = 1e-8
        multipliers[k] = (1.0 - e) / denom
    return multipliers


@njit(cache=True)
def _compute_moore_geom_corrections(spread_dir: float) -> np.ndarray:
    """
    对角度偏差 Δ ≤ 45° 的邻格乘以 1/cos(Δ) 校正因子，消除锯齿路径双重惩罚。
    """
    corrections = np.ones(8, dtype=np.float64)
    for k in range(8):
        diff = (_NB_ANGLES[k] - spread_dir) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        if diff <= 45.0:
            cos_val = np.cos(np.radians(diff))
            if cos_val > 1e-6:
                corrections[k] = 1.0 / cos_val
    return corrections


@njit(cache=True)
def _compute_direction_weights(spread_dir: float, power: float) -> np.ndarray:
    weights = np.empty(8, dtype=np.float64)
    total = 0.0
    for k in range(8):
        diff = abs(_NB_ANGLES[k] - spread_dir) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        cos_w = np.cos(np.radians(diff))
        if cos_w < 0.0:
            cos_w = 0.0
        w = cos_w ** power
        weights[k] = w
        total += w
    if total > 0.0:
        for k in range(8):
            weights[k] /= total
    else:
        for k in range(8):
            weights[k] = 0.125
    return weights


@njit(cache=True)
def _ca_step_active_fractional(
    state, burn_timer, ignition_progress, arrival_time, cumul_intensity,
    unspent_dt,                              # [Fix C] 未消耗动能池
    active_cells, active_count, ros, spread_dir, fli,
    pixel_size_m, dt, burn_duration, eccentricity, min_ros, current_time, save_intensity,
) -> Tuple:
    """
    """
    rows, cols = state.shape
    max_buf = active_count + active_count * 8 + 16
    pending   = np.empty((active_count * 8 + 16, 2), dtype=np.int32)
    pend_cnt  = 0
    new_active = np.empty((max_buf, 2), dtype=np.int32)
    new_count  = 0
    newly_ignited    = 0
    burned_out_count = 0

    for i in range(active_count):
        r = active_cells[i, 0]
        c = active_cells[i, 1]
        if state[r, c] != BURNING:
            continue
        cell_ros = float(ros[r, c])
        cell_dir = float(spread_dir[r, c])
        if cell_ros < min_ros:
            continue

        ros_mults = _compute_ellipse_ros_multipliers(cell_dir, eccentricity)

        geom_corr = _compute_moore_geom_corrections(cell_dir)

        effective_dt = dt + unspent_dt[r, c]
        unspent_dt[r, c] = 0.0

        for k in range(8):
            nr = r + _NB_OFFSETS_R[k]
            nc = c + _NB_OFFSETS_C[k]
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if state[nr, nc] != UNBURNED:
                continue

            neighbor_dist   = pixel_size_m * _NB_DIST_FAC[k]
            directional_ros = cell_ros * ros_mults[k] * geom_corr[k]

            prev_progress  = ignition_progress[nr, nc]
            progress_delta = (directional_ros * effective_dt) / neighbor_dist
            ignition_progress[nr, nc] += progress_delta

            if ignition_progress[nr, nc] >= 1.0 and prev_progress < 1.0:
                if progress_delta > 1e-12:
                    frac = (1.0 - prev_progress) / progress_delta
                    if frac < 0.0: frac = 0.0
                    if frac > 1.0: frac = 1.0
                else:
                    frac = 0.0

                residual_time = effective_dt * (1.0 - frac)   # 剩余未消耗时间

                state[nr, nc]             = BURNING
                arrival_time[nr, nc]      = current_time + frac * effective_dt
                burn_timer[nr, nc]        = residual_time     # [Fix A] 预置已燃时间
                unspent_dt[nr, nc]        = residual_time     # [Fix C] 存入残余动能池
                newly_ignited            += 1
                pending[pend_cnt, 0]      = nr
                pending[pend_cnt, 1]      = nc
                pend_cnt                 += 1

    for i in range(active_count):
        r = active_cells[i, 0]
        c = active_cells[i, 1]
        if state[r, c] != BURNING:
            continue
        if save_intensity:
            cumul_intensity[r, c] += fli[r, c] * dt
        burn_timer[r, c] += dt
        if burn_timer[r, c] >= burn_duration:
            has_pending_neighbor = False
            for k in range(8):
                nr = r + _NB_OFFSETS_R[k]
                nc = c + _NB_OFFSETS_C[k]
                if 0 <= nr < rows and 0 <= nc < cols:
                    if state[nr, nc] == UNBURNED and ignition_progress[nr, nc] > 0.0:
                        has_pending_neighbor = True
                        break
            if has_pending_neighbor:
                new_active[new_count, 0] = r
                new_active[new_count, 1] = c
                new_count += 1
            else:
                state[r, c]       = BURNED_OUT
                burned_out_count += 1
        else:
            new_active[new_count, 0] = r
            new_active[new_count, 1] = c
            new_count += 1

    for i in range(pend_cnt):
        r = pending[i, 0]
        c = pending[i, 1]
        if state[r, c] == BURNING:
            new_active[new_count, 0] = r
            new_active[new_count, 1] = c
            new_count += 1

    return new_active[:new_count].copy(), new_count, newly_ignited, burned_out_count


@njit(cache=True)
def _ca_step_active_threshold(
    state, burn_timer, arrival_time, cumul_intensity,
    active_cells, active_count, ros, spread_dir, fli,
    pixel_size_m, dt, burn_duration, direction_power, min_ros, base_prob,
    current_time, save_intensity,
) -> Tuple:
    rows, cols = state.shape
    max_buf   = active_count + active_count * 8 + 16
    pending   = np.empty((active_count * 8 + 16, 2), dtype=np.int32)
    pend_cnt  = 0
    new_active = np.empty((max_buf, 2), dtype=np.int32)
    new_count  = 0
    newly_ignited    = 0
    burned_out_count = 0

    for i in range(active_count):
        r = active_cells[i, 0]
        c = active_cells[i, 1]
        if state[r, c] != BURNING:
            continue
        cell_ros = float(ros[r, c])
        cell_dir = float(spread_dir[r, c])
        if cell_ros < min_ros:
            continue
        max_advance = cell_ros * dt
        weights = _compute_direction_weights(cell_dir, direction_power)
        for k in range(8):
            nr = r + _NB_OFFSETS_R[k]
            nc = c + _NB_OFFSETS_C[k]
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if state[nr, nc] != UNBURNED:
                continue
            neighbor_dist = pixel_size_m * _NB_DIST_FAC[k]
            if max_advance * weights[k] >= neighbor_dist * 0.5:
                if base_prob < 1.0:
                    if np.random.random() > base_prob:
                        continue
                state[nr, nc]        = BURNING
                arrival_time[nr, nc] = current_time + dt
                burn_timer[nr, nc]   = 0.0
                newly_ignited       += 1
                pending[pend_cnt, 0] = nr
                pending[pend_cnt, 1] = nc
                pend_cnt += 1

    for i in range(active_count):
        r = active_cells[i, 0]
        c = active_cells[i, 1]
        if state[r, c] != BURNING:
            continue
        if save_intensity:
            cumul_intensity[r, c] += fli[r, c] * dt
        burn_timer[r, c] += dt
        if burn_timer[r, c] >= burn_duration:
            state[r, c]       = BURNED_OUT
            burned_out_count += 1
        else:
            new_active[new_count, 0] = r
            new_active[new_count, 1] = c
            new_count += 1

    for i in range(pend_cnt):
        r = pending[i, 0]
        c = pending[i, 1]
        if state[r, c] == BURNING:
            new_active[new_count, 0] = r
            new_active[new_count, 1] = c
            new_count += 1

    return new_active[:new_count].copy(), new_count, newly_ignited, burned_out_count


def _interpolate_fire_behavior(beh_cur, beh_nxt, alpha):
    interp = {}
    interp['_profile'] = beh_cur.get('_profile')
    for key in ('rate_of_spread', 'fireline_intensity', 'flame_length', 'fire_type'):
        cur = beh_cur.get(key)
        nxt = beh_nxt.get(key)
        if cur is None:
            interp[key] = nxt
            continue
        if nxt is None:
            interp[key] = cur
            continue
        interp[key] = ((1.0 - alpha) * cur + alpha * nxt).astype(np.float32)
    cur_dir = beh_cur.get('spread_direction')
    nxt_dir = beh_nxt.get('spread_direction')
    if cur_dir is None:
        interp['spread_direction'] = nxt_dir
    elif nxt_dir is None:
        interp['spread_direction'] = cur_dir
    else:
        delta = nxt_dir - cur_dir
        delta = (delta + 180.0) % 360.0 - 180.0
        result = (cur_dir + alpha * delta) % 360.0
        interp['spread_direction'] = result.astype(np.float32)
    return interp


class RothermelOutputLoader:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def load_timestep(self, timestamp):
        ts_dir = self.output_dir / timestamp
        if not ts_dir.exists():
            return None
        files = {
            'rate_of_spread':     'rate_of_spread.tif',
            'spread_direction':   'spread_direction.tif',
            'flame_length':       'flame_length.tif',
            'fireline_intensity': 'fireline_intensity.tif',
            'fire_type':          'fire_type.tif',
        }
        data = {}
        profile = None
        for key, fname in files.items():
            fpath = ts_dir / fname
            if fpath.exists():
                with rasterio.open(str(fpath)) as src:
                    arr = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        arr = np.where(arr == src.nodata, np.nan, arr)
                    data[key] = arr
                    if profile is None:
                        profile = src.profile.copy()
            else:
                print(f"  警告: 缺少 {fpath}")
        data['_profile'] = profile
        return data

    def discover_timestamps(self):
        timestamps = []
        for entry in os.listdir(self.output_dir):
            entry_path = self.output_dir / entry
            if entry_path.is_dir():
                try:
                    datetime.strptime(entry, "%Y%m%d_%H%M")
                    timestamps.append(entry)
                except ValueError:
                    continue
        timestamps.sort()
        return timestamps

    def get_reference_profile(self, timestamps):
        for ts in timestamps:
            fpath = self.output_dir / ts / 'rate_of_spread.tif'
            if fpath.exists():
                with rasterio.open(str(fpath)) as src:
                    return src.profile.copy()
        raise FileNotFoundError("无法找到任何有效的栅格文件")


class FireCAState:
    _INIT_CAPACITY = 4096

    def __init__(self, shape, fuel_model=None):
        self.shape = shape
        self.state                = np.full(shape, UNBURNED, dtype=np.int8)
        self.burn_timer           = np.zeros(shape, dtype=np.float32)
        self.arrival_time         = np.full(shape, np.nan, dtype=np.float32)
        self.cumulative_intensity = np.zeros(shape, dtype=np.float32)
        self.ignition_progress    = np.zeros(shape, dtype=np.float32)
        # [Fix C] 未消耗动能池
        self.unspent_dt           = np.zeros(shape, dtype=np.float32)

        self._active_cap  = self._INIT_CAPACITY
        self.active_cells = np.zeros((self._active_cap, 2), dtype=np.int32)
        self.active_count = 0

        if fuel_model is not None:
            nb_mask = (
                np.isnan(fuel_model) |
                np.isin(fuel_model.astype(int), [91, 92, 93, 98, 99])
            )
            self.state[nb_mask] = NON_BURNABLE

        self.current_time = 0.0
        self.step_count   = 0

    def _append_active(self, row, col):
        if self.active_count >= self._active_cap:
            new_cap = self._active_cap * 2
            new_buf = np.zeros((new_cap, 2), dtype=np.int32)
            new_buf[:self.active_count] = self.active_cells[:self.active_count]
            self.active_cells  = new_buf
            self._active_cap   = new_cap
        self.active_cells[self.active_count, 0] = row
        self.active_cells[self.active_count, 1] = col
        self.active_count += 1

    def ignite(self, row, col, current_time=0.0):
        if self.state[row, col] == UNBURNED:
            self.state[row, col]             = BURNING
            self.arrival_time[row, col]      = current_time
            self.burn_timer[row, col]        = 0.0
            self.ignition_progress[row, col] = 0.0
            self.unspent_dt[row, col]        = 0.0
            self._append_active(row, col)

    def ignite_region(self, mask, current_time=0.0):
        ignitable = (self.state == UNBURNED) & mask
        rows_idx, cols_idx = np.where(ignitable)
        self.state[ignitable]             = BURNING
        self.arrival_time[ignitable]      = current_time
        self.burn_timer[ignitable]        = 0.0
        self.ignition_progress[ignitable] = 0.0
        self.unspent_dt[ignitable]        = 0.0
        for r, c in zip(rows_idx.tolist(), cols_idx.tolist()):
            self._append_active(int(r), int(c))

    def update_active_list(self, new_active, new_count):
        if new_count > self._active_cap:
            self._active_cap  = max(new_count * 2, self._INIT_CAPACITY)
            self.active_cells = np.zeros((self._active_cap, 2), dtype=np.int32)
        self.active_cells[:new_count] = new_active
        self.active_count = new_count

    @property
    def burning_count(self):
        return self.active_count

    @property
    def burned_count(self):
        return int(np.sum(self.state == BURNED_OUT))

    @property
    def unburned_count(self):
        return int(np.sum(self.state == UNBURNED))


class FireSpreadCA:
    def __init__(self, config):
        self.config = config

        # ------------------------------------------------------------------ #
        # ------------------------------------------------------------------ #
        if config.eccentricity is not None:
            self._eccentricity = float(config.eccentricity)
            # 反推 back_to_head_ratio 仅用于日志/元数据的一致性
            e = self._eccentricity
            config.back_to_head_ratio = (1.0 - e) / (1.0 + e)
            print(f"  [Fix D] 使用显式偏心率 e={self._eccentricity:.3f}  "
                  f"(等效 back/head={config.back_to_head_ratio:.3f})")
        else:
            r = config.back_to_head_ratio
            self._eccentricity = (1.0 - r) / (1.0 + r)
            print(f"  [Fix D] 由 back_to_head_ratio={r} 推算 e={self._eccentricity:.3f}")

        self._jit_warmed_up = False

    def _warmup_jit(self):
        if not NUMBA_AVAILABLE or self._jit_warmed_up:
            return
        print("  [Numba] 首次 JIT 编译中，请稍等（约 15~40 秒）...")
        tiny = (12, 12)
        _s   = np.zeros(tiny, dtype=np.int8)
        _bt  = np.zeros(tiny, dtype=np.float32)
        _ip  = np.zeros(tiny, dtype=np.float32)
        _at  = np.full(tiny, np.nan, dtype=np.float32)
        _ci  = np.zeros(tiny, dtype=np.float32)
        _udt = np.zeros(tiny, dtype=np.float32)
        _ros = np.ones(tiny,  dtype=np.float32)
        _sd  = np.zeros(tiny, dtype=np.float32)
        _fli = np.zeros(tiny, dtype=np.float32)
        _s[6, 6] = BURNING
        _bt[6, 6] = 0.0
        _ac = np.array([[6, 6]], dtype=np.int32)
        _ca_step_active_fractional(
            _s, _bt, _ip, _at, _ci, _udt, _ac, 1,
            _ros, _sd, _fli, 30., 1., 90., self._eccentricity, 0.01, 0., True
        )
        _ca_step_active_threshold(
            _s, _bt, _at, _ci, _ac, 1,
            _ros, _sd, _fli, 30., 1., 90., 2., 0.01, 1., 0., True
        )
        self._jit_warmed_up = True
        print("  [Numba] JIT 编译完成 ✓")

    def step(self, ca_state, fire_behavior, pixel_size_m, fuel_model=None, dt_override=None):
        self._warmup_jit()
        if ca_state.active_count == 0:
            ca_state.current_time += self.config.dt_minutes
            ca_state.step_count   += 1
            return {
                'step': ca_state.step_count, 'time_minutes': ca_state.current_time,
                'newly_ignited': 0, 'burned_out': 0,
                'burning': 0, 'total_burned': ca_state.burned_count,
            }
        dt    = dt_override if dt_override is not None else self.config.dt_minutes
        shape = ca_state.state.shape
        ros        = fire_behavior.get('rate_of_spread',    np.zeros(shape, dtype=np.float32))
        spread_dir = fire_behavior.get('spread_direction',  np.zeros(shape, dtype=np.float32))
        fli        = fire_behavior.get('fireline_intensity', np.zeros(shape, dtype=np.float32))
        ros        = np.where(np.isnan(ros)        | (ros < 0), 0., ros).astype(np.float32)
        spread_dir = np.where(np.isnan(spread_dir),             0., spread_dir).astype(np.float32)
        fli        = np.where(np.isnan(fli)        | (fli < 0), 0., fli).astype(np.float32)

        # 单位转换：m/s → m/min
        if self.config.ros_unit == 'm_per_s':
            ros = (ros * 60.0).astype(np.float32)

        # ------------------------------------------------------------------ #
        # ------------------------------------------------------------------ #
        if self.config.ros_scale_factor != 1.0:
            ros = (ros * self.config.ros_scale_factor).astype(np.float32)

        ac_slice = np.ascontiguousarray(ca_state.active_cells[:ca_state.active_count])

        if self.config.use_fractional_accumulation:
            new_ac, new_cnt, newly_ignited, burned_out = _ca_step_active_fractional(
                ca_state.state, ca_state.burn_timer,
                ca_state.ignition_progress,
                ca_state.arrival_time, ca_state.cumulative_intensity,
                ca_state.unspent_dt,
                ac_slice, ca_state.active_count,
                ros, spread_dir, fli,
                pixel_size_m, dt,
                self.config.burn_duration_minutes,
                self._eccentricity,           # [Fix D] 动态偏心率
                self.config.min_ros_threshold,
                ca_state.current_time,
                self.config.save_intensity_map,
            )
        else:
            new_ac, new_cnt, newly_ignited, burned_out = _ca_step_active_threshold(
                ca_state.state, ca_state.burn_timer,
                ca_state.arrival_time, ca_state.cumulative_intensity,
                ac_slice, ca_state.active_count,
                ros, spread_dir, fli,
                pixel_size_m, dt,
                self.config.burn_duration_minutes,
                self.config.direction_weight_power,
                self.config.min_ros_threshold,
                self.config.base_ignition_prob,
                ca_state.current_time,
                self.config.save_intensity_map,
            )

        ca_state.update_active_list(new_ac, int(new_cnt))
        ca_state.current_time += dt
        ca_state.step_count   += 1

        return {
            'step':          ca_state.step_count,
            'time_minutes':  ca_state.current_time,
            'newly_ignited': int(newly_ignited),
            'burned_out':    int(burned_out),
            'burning':       ca_state.active_count,
            'total_burned':  ca_state.burned_count,
        }


class CAOutputWriter:
    def __init__(self, output_dir, reference_profile):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile = reference_profile

    def _write_tif(self, arr, fpath, dtype, nodata):
        profile = {**self.profile, 'dtype': dtype, 'nodata': nodata}
        with rasterio.open(str(fpath), 'w', **profile) as dst:
            dst.write(arr.astype(dtype), 1)

    def save_state_snapshot(self, state, step, time_minutes):
        subdir = self.output_dir / 'snapshots'
        subdir.mkdir(exist_ok=True)
        fpath = subdir / f'fire_state_step{step:05d}_t{int(time_minutes):06d}min.tif'
        self._write_tif(state, fpath, 'int8', -1)

    def save_arrival_time(self, arrival_time):
        fpath = self.output_dir / 'arrival_time_minutes.tif'
        arr = np.where(np.isnan(arrival_time), -9999, arrival_time).astype('float32')
        self._write_tif(arr, fpath, 'float32', -9999)
        print(f"  ✓ 到达时间图:       {fpath}")

    def save_final_state(self, state):
        fpath = self.output_dir / 'fire_perimeter_final.tif'
        self._write_tif(state, fpath, 'int8', -1)
        print(f"  ✓ 最终状态图:       {fpath}")

    def save_ignition_progress(self, progress):
        fpath = self.output_dir / 'ignition_progress.tif'
        self._write_tif(progress.astype('float32'), fpath, 'float32', -9999)
        print(f"  ✓ 点燃进度图:       {fpath}")

    def save_cumulative_intensity(self, intensity):
        fpath = self.output_dir / 'cumulative_fireline_intensity.tif'
        arr = np.where(intensity <= 0, -9999, intensity).astype('float32')
        self._write_tif(arr, fpath, 'float32', -9999)
        print(f"  ✓ 累积火线强度图:   {fpath}")

    def save_statistics_csv(self, stats_list):
        fpath = self.output_dir / 'ca_simulation_statistics.csv'
        if not stats_list:
            return
        with open(str(fpath), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(stats_list[0].keys()))
            writer.writeheader()
            writer.writerows(stats_list)
        print(f"  ✓ 统计CSV:          {fpath}")


def create_ignition_mask_from_geojson(geojson_path, reference_profile, shape):
    try:
        from rasterio.features import rasterize
        from shapely.geometry import shape as shapely_shape
        with open(geojson_path, 'r') as f:
            gj = json.load(f)
        geometries = [
            (shapely_shape(feat['geometry']), 1)
            for feat in gj.get('features', [])
            if feat.get('geometry')
        ]
        if not geometries:
            raise ValueError("GeoJSON 中没有有效几何体")
        mask = rasterize(
            geometries,
            out_shape=shape,
            transform=reference_profile['transform'],
            fill=0, dtype='uint8'
        )
        return mask.astype(bool)
    except ImportError:
        print("  警告: 需要安装 shapely 才能从 GeoJSON 创建点火区域")
        return np.zeros(shape, dtype=bool)


def run_ca_fire_simulation(
    rothermel_output_dir, ca_output_dir,
    ignition_points=None, ignition_geojson=None,
    config=None, fuel_model_path=None, timestamps=None,
):
    if config is None:
        config = CAConfig()

    r_bh  = config.back_to_head_ratio
    e_val = config.eccentricity if config.eccentricity is not None else (1.0 - r_bh) / (1.0 + r_bh)

    print("=" * 70)
    print("CA 森林火灾蔓延模拟 V8.0 (Fix A~E)")
    print("=" * 70)
    print(f"  [Fix A] 亚步点燃时刻修正:        启用（消除每格 ~0.5×dt 罚站偏慢）")
    print(f"  [Fix B] Moore 几何量化补偿:      启用（1/cos(Δ) 校正，Δ≤45° 邻格）")
    print(f"  [Fix C] 闭环连续时间积分:        启用（unspent_dt 池）")
    print(f"  [Fix D] 椭圆偏心率解锁:          "
          f"e={e_val:.3f}  "
          f"({'显式指定' if config.eccentricity is not None else f'由 back/head={r_bh} 推算'})")
    print(f"  [Fix E] ROS 标定因子:            ×{config.ros_scale_factor:.3f}  "
          f"({'已启用校正' if config.ros_scale_factor != 1.0 else '默认无修正'})")
    print(f"  分数阶累积:   {'启用' if config.use_fractional_accumulation else '禁用（兼容模式）'}")
    print(f"  ROS 单位:     {config.ros_unit}")
    print(f"  Numba JIT:    {'可用' if NUMBA_AVAILABLE else '不可用（纯Python）'}")
    print(f"  气象时空平滑: {'启用（线性插值）' if config.use_meteo_interpolation else '禁用（阶跃切换）'}")
    print(f"  自适应微步:   {'启用' if config.use_adaptive_timestep else '禁用'}  "
          f"max_CFL={config.max_cfl}")

    loader = RothermelOutputLoader(rothermel_output_dir)
    if timestamps is None:
        timestamps = loader.discover_timestamps()
    if not timestamps:
        raise ValueError(f"在 {rothermel_output_dir} 中未找到时间步数据")

    print(f"\n发现 {len(timestamps)} 个时间步: {timestamps[0]} → {timestamps[-1]}")

    ref_profile = loader.get_reference_profile(timestamps)
    shape       = (ref_profile['height'], ref_profile['width'])

    if config.pixel_size_m is not None:
        pixel_size = config.pixel_size_m
    else:
        pixel_size = abs(ref_profile['transform'].a)
        if pixel_size < 1.0:
            pixel_size = pixel_size * 111000
            print(f"  警告: 检测到地理坐标，像素大小约 {pixel_size:.0f}m")
    print(f"像素分辨率: {pixel_size:.1f} m   |   栅格大小: {shape[0]}×{shape[1]} = {shape[0]*shape[1]:,} 像素")

    fuel_model = None
    if fuel_model_path and os.path.exists(fuel_model_path):
        with rasterio.open(fuel_model_path) as src:
            fm_shape = (src.height, src.width)
            if fm_shape == shape:
                fuel_model = src.read(1).astype(np.float32)
            else:
                fuel_model = np.full(shape, np.nan, dtype=np.float32)
                rasterio.warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=fuel_model,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_profile['transform'],
                    dst_crs=ref_profile['crs'],
                    resampling=rasterio.warp.Resampling.nearest,
                )
                print(f"  ⚠ 燃料模型尺寸 {fm_shape} ≠ 参考栅格 {shape}，"
                      f"已自动重采样对齐 ✓")
        print(f"燃料模型已加载: {fuel_model_path}")

    ca_state = FireCAState(shape, fuel_model)

    if ignition_geojson and os.path.exists(ignition_geojson):
        mask = create_ignition_mask_from_geojson(ignition_geojson, ref_profile, shape)
        ca_state.ignite_region(mask, current_time=0.0)
        print(f"点火区域 (GeoJSON): {int(np.sum(mask))} 个像素")
    elif ignition_points:
        for row, col in ignition_points:
            ca_state.ignite(row, col, current_time=0.0)
        print(f"点火点: {len(ignition_points)} 个像素")
    else:
        cr, cc = shape[0] // 2, shape[1] // 2
        ca_state.ignite(cr, cc, current_time=0.0)
        print(f"默认点火点: 中心像素 ({cr}, {cc})")

    print(f"初始活动队列长度: {ca_state.active_count}")

    engine = FireSpreadCA(config)
    writer = CAOutputWriter(ca_output_dir, ref_profile)

    total_sim_time_minutes = len(timestamps) * 60.0
    total_ca_steps         = int(total_sim_time_minutes / config.dt_minutes)
    minutes_per_meteo_step = 60.0

    print(f"\n模拟参数:")
    print(f"  CA 宏观时间步长   : {config.dt_minutes} 分钟")
    print(f"  燃烧持续时间      : {config.burn_duration_minutes} 分钟")
    print(f"  总模拟时间        : {total_sim_time_minutes:.0f} min ({total_sim_time_minutes/60:.1f} h)")
    print(f"  总 CA 步数        : {total_ca_steps:,}")
    print(f"  椭圆偏心率        : {e_val:.3f}  (back/head={config.back_to_head_ratio:.3f})")
    print(f"  ROS 标定因子      : ×{config.ros_scale_factor:.3f}")
    print(f"  CFL 上限          : {config.max_cfl}")
    print("\n开始模拟...\n")

    cached_idx_cur = -1
    beh_cur = None
    beh_nxt = None

    def _ensure_cached(meteo_idx):
        nonlocal cached_idx_cur, beh_cur, beh_nxt
        if cached_idx_cur == meteo_idx:
            return
        beh_cur = loader.load_timestep(timestamps[meteo_idx])
        nxt_idx = meteo_idx + 1
        if nxt_idx < len(timestamps):
            beh_nxt = loader.load_timestep(timestamps[nxt_idx])
        else:
            beh_nxt = None
        cached_idx_cur = meteo_idx

    all_stats    = []
    ts_sim_start = time_module.time()
    step_times   = []
    total_substeps_taken = 0

    for step_i in range(total_ca_steps):
        current_sim_minutes = step_i * config.dt_minutes
        meteo_idx = min(
            int(current_sim_minutes / minutes_per_meteo_step),
            len(timestamps) - 1
        )
        _ensure_cached(meteo_idx)

        if beh_cur is None:
            ca_state.current_time += config.dt_minutes
            ca_state.step_count   += 1
            continue

        if config.use_meteo_interpolation and beh_nxt is not None:
            alpha = (current_sim_minutes % minutes_per_meteo_step) / minutes_per_meteo_step
            fire_behavior = _interpolate_fire_behavior(beh_cur, beh_nxt, alpha)
        else:
            alpha = 0.0
            fire_behavior = beh_cur

        if config.verbose and (
            step_i == 0 or (
                int(current_sim_minutes / minutes_per_meteo_step) !=
                int((current_sim_minutes - config.dt_minutes) / minutes_per_meteo_step)
            )
        ):
            ts = timestamps[meteo_idx]
            nxt_ts = timestamps[meteo_idx + 1] if meteo_idx + 1 < len(timestamps) else "—"
            interp_flag = f"α={alpha:.2f}" if config.use_meteo_interpolation else "阶跃"
            print(f"  [t={current_sim_minutes:.0f}min] 气象帧 {ts}→{nxt_ts}  "
                  f"插值={interp_flag}  活动队列={ca_state.active_count:,}")

        t0 = time_module.perf_counter()
        n_sub = 1
        dt_sub = config.dt_minutes

        if config.use_adaptive_timestep and ca_state.active_count > 0:
            ac_rows = ca_state.active_cells[:ca_state.active_count, 0]
            ac_cols = ca_state.active_cells[:ca_state.active_count, 1]
            ros_field = fire_behavior.get('rate_of_spread')
            if ros_field is not None:
                active_ros = ros_field[ac_rows, ac_cols]
                max_ros = float(np.nanmax(active_ros))
                if config.ros_unit == 'm_per_s':
                    max_ros *= 60.0
                # [Fix E] CFL 计算中同样考虑 ros_scale_factor
                max_ros *= config.ros_scale_factor
                cfl = max_ros * config.dt_minutes / pixel_size
                if cfl > config.max_cfl and max_ros > 0:
                    n_sub = int(np.ceil(cfl / config.max_cfl))
                    dt_sub = config.dt_minutes / n_sub

        total_substeps_taken += n_sub
        step_stats = None

        for sub_i in range(n_sub):
            sub_stats = engine.step(
                ca_state=ca_state,
                fire_behavior=fire_behavior,
                pixel_size_m=pixel_size,
                dt_override=dt_sub,
            )
            step_stats = sub_stats

        step_dur = time_module.perf_counter() - t0
        step_times.append(step_dur)

        total_burned_px = ca_state.active_count + ca_state.burned_count
        step_stats['total_burned_ha'] = total_burned_px * (pixel_size ** 2) / 10000
        step_stats['meteo_timestamp'] = timestamps[meteo_idx]
        step_stats['meteo_alpha']     = round(alpha, 4)
        step_stats['active_cells']    = ca_state.active_count
        step_stats['n_substeps']      = n_sub
        step_stats['dt_sub_minutes']  = round(dt_sub, 4)
        all_stats.append(step_stats)

        if step_i % config.save_state_every_n_steps == 0:
            writer.save_state_snapshot(ca_state.state, step_i, ca_state.current_time)
            elapsed  = time_module.time() - ts_sim_start
            progress = (step_i + 1) / total_ca_steps * 100
            avg_ms   = np.mean(step_times[-60:]) * 1000 if step_times else 0
            avg_sub  = total_substeps_taken / max(step_i + 1, 1)
            print(f"  步 {step_i+1:5d}/{total_ca_steps} | "
                  f"t={ca_state.current_time:.0f}min | "
                  f"活动={ca_state.active_count:6,} | "
                  f"燃尽={ca_state.burned_count:7,} | "
                  f"面积={step_stats['total_burned_ha']:8.1f}ha | "
                  f"亚步={n_sub}(均{avg_sub:.1f}) | "
                  f"步耗={avg_ms:.1f}ms | "
                  f"总耗={elapsed:.1f}s | {progress:.1f}%")

        if ca_state.active_count == 0:
            print(f"\n  火已熄灭，在第 {step_i+1} 步 (t={ca_state.current_time:.1f}min)")
            break

    print("\n保存最终结果...")
    writer.save_final_state(ca_state.state)
    if config.save_arrival_time:
        writer.save_arrival_time(ca_state.arrival_time)
    if config.save_intensity_map:
        writer.save_cumulative_intensity(ca_state.cumulative_intensity)
    if config.use_fractional_accumulation:
        writer.save_ignition_progress(ca_state.ignition_progress)
    writer.save_statistics_csv(all_stats)
    writer.save_state_snapshot(ca_state.state, ca_state.step_count, ca_state.current_time)

    total_elapsed   = time_module.time() - ts_sim_start
    total_burned_px = ca_state.active_count + ca_state.burned_count
    final_area_ha   = total_burned_px * (pixel_size ** 2) / 10000
    avg_step_ms     = np.mean(step_times) * 1000 if step_times else 0

    metadata = {
        'version': 'CA-FireSpread-V8.0',
        'improvements': [
            'Fix A (V6.0): Sub-step ignition timing.',
            'Fix B (V6.0): Moore 8-neighbourhood geometric correction.',
            'Fix C (V7.0): Continuous time integration via unspent_dt pool.',
            'Fix D (V8.0): Explicit eccentricity parameter unlocked in CAConfig. '
            'Users can now set eccentricity directly (e.g. 0.75 for strong-wind fire) '
            'instead of being constrained to back_to_head_ratio derivation.',
            'Fix E (V8.0): ROS scale factor (ros_scale_factor) introduced as SRAF '
            'substitute. Applied after unit conversion in FireSpreadCA.step() and '
            'also fed into CFL adaptive timestep calculation for consistency.',
        ],
        'config': {
            'dt_minutes':                  config.dt_minutes,
            'burn_duration_minutes':       config.burn_duration_minutes,
            'pixel_size_m':                pixel_size,
            'ros_unit':                    config.ros_unit,
            'ros_scale_factor':            config.ros_scale_factor,
            'use_fractional_accumulation': config.use_fractional_accumulation,
            'use_meteo_interpolation':     config.use_meteo_interpolation,
            'back_to_head_ratio':          config.back_to_head_ratio,
            'ellipse_eccentricity':        round(e_val, 4),
            'eccentricity_source':         'explicit' if config.eccentricity is not None else 'derived',
            'use_adaptive_timestep':       config.use_adaptive_timestep,
            'max_cfl':                     config.max_cfl,
        },
        'simulation_summary': {
            'total_steps':                ca_state.step_count,
            'total_time_minutes':         ca_state.current_time,
            'total_elapsed_seconds':      total_elapsed,
            'avg_step_ms':                round(avg_step_ms, 3),
            'final_burned_area_ha':       final_area_ha,
            'burning_pixels':             ca_state.active_count,
            'burned_out_pixels':          ca_state.burned_count,
            'total_substeps_taken':       total_substeps_taken,
            'avg_substeps_per_macrostep': round(total_substeps_taken / max(ca_state.step_count, 1), 2),
            'numba_available':            NUMBA_AVAILABLE,
        }
    }
    with open(os.path.join(ca_output_dir, 'ca_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("CA 模拟完成!")
    print("=" * 70)
    print(f"  总步数          : {ca_state.step_count:,}")
    print(f"  模拟时长        : {ca_state.current_time:.0f} min ({ca_state.current_time/60:.1f} h)")
    print(f"  最终过火面积    : {final_area_ha:.1f} 公顷")
    print(f"  平均每步耗时    : {avg_step_ms:.2f} ms")
    print(f"  总亚步数        : {total_substeps_taken:,}  "
          f"(均 {total_substeps_taken/max(ca_state.step_count,1):.1f} sub/macro)")
    print(f"  总运行时间      : {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"\n  输出目录: {ca_output_dir}")
    print("=" * 70)

    return os.path.join(ca_output_dir, 'ca_simulation_statistics.csv')


def main():
    ROTHERMEL_OUTPUT_DIR = r"E:\Task\fire_simulation\fire_behavior_output_batch_utm"
    STATIC_DATA_DIR      = r"D:\Hly\Data\ERA5\xichang20200330\terrain"
    CA_OUTPUT_DIR        = r"E:\Task\fire_simulation\ca_spread_output"
    FUEL_MODEL_PATH      = os.path.join(STATIC_DATA_DIR, "FuelModel_Xichang_2019.tif")

    if not os.path.exists(FUEL_MODEL_PATH):
        print(f"⚠ 警告: 燃料模型文件未找到: {FUEL_MODEL_PATH}")
    else:
        print(f"✓ 燃料模型文件已确认: {FUEL_MODEL_PATH}")

    print("远程连接成功")

    IGNITION_POINTS  = [(890, 917)]
    IGNITION_GEOJSON = None

    config = CAConfig(
        dt_minutes                  = 1.0,
        burn_duration_minutes       = 90.0,
        base_ignition_prob          = 0.95,

        # ---------------------------------------------------------------- #
        # ---------------------------------------------------------------- #
        eccentricity                = 0.75,   # [Fix D] 新增，替代原 back_to_head_ratio=0.2

        back_to_head_ratio          = 0.2,    # 当 eccentricity 已指定时仅作元数据记录

        ros_unit                    = 'm_per_min',

        # ---------------------------------------------------------------- #
        # ---------------------------------------------------------------- #
        ros_scale_factor            = 1.0,    # [Fix E] 新增；按需调整

        use_fractional_accumulation = True,
        use_meteo_interpolation     = True,
        use_adaptive_timestep       = True,
        max_cfl                     = 0.9,
        min_ros_threshold           = 0.05,
        save_state_every_n_steps    = 60,
        save_arrival_time           = True,
        save_intensity_map          = True,
        verbose                     = True,
    )

    TIMESTAMPS = None

    try:
        run_ca_fire_simulation(
            rothermel_output_dir = ROTHERMEL_OUTPUT_DIR,
            ca_output_dir        = CA_OUTPUT_DIR,
            ignition_points      = IGNITION_POINTS,
            ignition_geojson     = IGNITION_GEOJSON,
            config               = config,
            fuel_model_path      = FUEL_MODEL_PATH,
            timestamps           = TIMESTAMPS,
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


def geo_to_pixel(lon, lat, raster_path):
    with rasterio.open(raster_path) as src:
        row, col = src.index(lon, lat)
        return int(row), int(col)


if __name__ == "__main__":
    main()