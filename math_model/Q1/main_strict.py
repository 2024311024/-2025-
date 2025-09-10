
"""
严格遮蔽模型：烟幕干扰弹投放策略
使用严格遮蔽判定计算烟幕云对导弹的遮蔽效果
"""
import json, math, os, sys, logging
import numpy as np
from typing import List, Tuple


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

from utils_geometry import unit, vec_add, vec_scale, dist_point_to_segment


def missile_pos(M0, v_m, t):
    """计算导弹在t时刻的位置"""
    u = unit((-M0[0], -M0[1], -M0[2]))  # toward origin
    return vec_add(M0, vec_scale(u, v_m*t))

def uav_pos(FY1_0, v_uav, heading, t):
    """计算无人机在t时刻的位置"""
    return (FY1_0[0] + v_uav*heading[0]*t,
            FY1_0[1] + v_uav*heading[1]*t,
            FY1_0[2] + v_uav*heading[2]*t)

def bomb_pos_at_burst(cfg):
    t = cfg["delay_to_burst"]
    R = uav_pos(tuple(cfg["uav"]["FY1_0"]), cfg["uav"]["speed"], tuple(cfg["uav"]["heading"]), cfg["t_release"])
    vx = cfg["uav"]["speed"]*cfg["uav"]["heading"][0]
    vy = cfg["uav"]["speed"]*cfg["uav"]["heading"][1]
    vz0 = 0.0
    x = R[0] + vx*t
    y = R[1] + vy*t
    z = R[2] + vz0*t - 0.5*cfg["g"]*t*t
    return (x,y,z)

def cloud_center(cfg, D, t):
    dt = t - (cfg["t_release"] + cfg["delay_to_burst"])
    return (D[0], D[1], D[2] - cfg["sink_speed"]*dt)

def build_series(cfg):
    M0 = tuple(cfg["missile"]["M0"])
    v_m = cfg["missile"]["speed"]
    T = tuple(cfg["true_target"]["T"])
    t_burst = cfg["t_release"] + cfg["delay_to_burst"]
    D = bomb_pos_at_burst(cfg)
    times = []
    Ms = []
    Cs = []
    rs = []
    t = t_burst
    t_end = t_burst + cfg["effective_after_burst"]
    dt = cfg["delta_t"]
    while t <= t_end + 1e-12:
        times.append(t)
        Ms.append(missile_pos(M0, v_m, t))
        Cs.append(cloud_center(cfg, D, t))
        rs.append(cfg["radius"])
        t += dt
    return times, Ms, T, Cs, rs

def solve_scan_strict(cfg):
    """使用严格遮蔽判定的扫描算法"""
    times, Ms, T, Cs, rs = build_series(cfg)
    
    # 参数占位（若未来扩展圆柱模型）；当前未使用，避免未使用变量
    _ = cfg.get("cylinder_height", 10.0)
    
    def is_blocked_at_time(t):
        # 插值计算导弹位置和烟幕中心
        alpha = (t - times[0]) / (times[-1] - times[0]) if len(times) > 1 else 0
        alpha = max(0, min(1, alpha))
        
        # 找到最近的时间索引
        idx = min(int(alpha * (len(times) - 1)), len(times) - 2)
        local_alpha = alpha * (len(times) - 1) - idx
        local_alpha = max(0, min(1, local_alpha))
        
        M = (Ms[idx][0]*(1-local_alpha) + Ms[idx+1][0]*local_alpha,
             Ms[idx][1]*(1-local_alpha) + Ms[idx+1][1]*local_alpha,
             Ms[idx][2]*(1-local_alpha) + Ms[idx+1][2]*local_alpha)
        C = (Cs[idx][0]*(1-local_alpha) + Cs[idx+1][0]*local_alpha,
             Cs[idx][1]*(1-local_alpha) + Cs[idx+1][1]*local_alpha,
             Cs[idx][2]*(1-local_alpha) + Cs[idx+1][2]*local_alpha)
        
        # 严格遮蔽判定：距离必须严格小于半径
        dist, lam = dist_point_to_segment(C, M, T)
        strict_radius = cfg["radius"] - 0.1  # 减少0.1米容差
        is_blocked = dist < strict_radius
        return is_blocked
    
    # 使用二分法寻找遮蔽区间的边界
    events = []
    if len(times) == 0:
        return 0.0, [], 0.0, 0.0
    if len(times) == 1:
        # 单点评估
        single_blocked = is_blocked_at_time(times[0])
        return (cfg.get("delta_t", 0.0) if single_blocked else 0.0), ([(times[0], times[0])] if single_blocked else []), times[0], times[0]

    t_start = times[0]
    t_end = times[-1]
    dt = cfg["delta_t"]
    
    # 扫描寻找状态变化点
    prev_blocked = is_blocked_at_time(t_start)
    t = t_start + dt
    
    while t <= t_end:
        curr_blocked = is_blocked_at_time(t)
        if prev_blocked != curr_blocked:
            # 状态变化，使用二分法精确定位
            lo, hi = t - dt, t
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                mid_blocked = is_blocked_at_time(mid)
                if mid_blocked == prev_blocked:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-12:
                    break
            events.append(0.5 * (lo + hi))
        prev_blocked = curr_blocked
        t += dt
    
    events.sort()
    intervals = []
    total = 0.0
    
    # 构建遮蔽区间
    for i in range(0, len(events), 2):
        if i+1 < len(events):
            intervals.append((events[i], events[i+1]))
            total += events[i+1]-events[i]
    
    return total, intervals, times[0], times[-1]

def main():
    cfg_path = os.path.join(PROJ_ROOT, "config.json")
    cfg = json.load(open(cfg_path, "r"))
    
    # 严格遮蔽模型
    total_strict, intervals_strict, _, _ = solve_scan_strict(cfg)
    print(f"严格遮蔽模型: {total_strict:.6f} 秒, 区间: {intervals_strict}")
    
    # 球体模型对比（同目录模块，通过包方式或路径已加入实现导入）
    from Q1.method_A_miqcp import solve_scan as solve_scan
    total_sphere, intervals_sphere, _, _ = solve_scan(cfg)
    print(f"球体模型: {total_sphere:.6f} 秒, 区间: {intervals_sphere}")

if __name__ == "__main__":
    main()
