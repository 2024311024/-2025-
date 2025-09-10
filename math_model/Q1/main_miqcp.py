
"""
球体模型：烟幕干扰弹投放策略
使用球体模型计算烟幕云对导弹的遮蔽效果
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
    u = unit((-M0[0], -M0[1], -M0[2]))
    return vec_add(M0, vec_scale(u, v_m*t))

def uav_pos(FY1_0, v_uav, heading, t):
    """计算无人机在t时刻的位置"""
    return (FY1_0[0] + v_uav*heading[0]*t,
            FY1_0[1] + v_uav*heading[1]*t,
            FY1_0[2] + v_uav*heading[2]*t)

def bomb_pos_at_burst(cfg):
    """计算烟幕弹起爆时的位置"""
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
    """计算烟幕云中心在t时刻的位置"""
    dt = t - (cfg["t_release"] + cfg["delay_to_burst"])
    return (D[0], D[1], D[2] - cfg["sink_speed"]*dt)

def build_series(cfg):
    """构建时间序列数据，包含导弹位置、烟幕云中心等"""
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

def solve_scan(cfg):
    """使用精确扫描和二分法计算遮蔽区间和目标函数值"""
    from utils_geometry import dist_point_to_segment
    times, Ms, T, Cs, rs = build_series(cfg)
    f = [dist_point_to_segment(Cs[i], Ms[i], T)[0] - rs[i] for i in range(len(times))]
    events = []
    for i in range(1, len(times)):
        if f[i-1]*f[i] < 0:
            lo, hi = times[i-1], times[i]
            flo, fhi = f[i-1], f[i]
            for _ in range(60):
                mid = 0.5*(lo+hi)
                # interpolate linearly in time for mid-eval
                alpha = (mid - times[i-1])/(times[i]-times[i-1])
                Mm = (Ms[i-1][0]*(1-alpha)+Ms[i][0]*alpha,
                      Ms[i-1][1]*(1-alpha)+Ms[i][1]*alpha,
                      Ms[i-1][2]*(1-alpha)+Ms[i][2]*alpha)
                Cm = (Cs[i-1][0]*(1-alpha)+Cs[i][0]*alpha,
                      Cs[i-1][1]*(1-alpha)+Cs[i][1]*alpha,
                      Cs[i-1][2]*(1-alpha)+Cs[i][2]*alpha)
                fm = dist_point_to_segment(Cm, Mm, T)[0] - cfg["radius"]
                if flo*fm <= 0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
            events.append(0.5*(lo+hi))
    events.sort()
    intervals = []
    total = 0.0
    for i in range(0, len(events), 2):
        if i+1 < len(events):
            intervals.append((events[i], events[i+1]))
            total += events[i+1]-events[i]
    return total, intervals, times[0], times[-1]

def setup_logging():
    """设置日志记录"""
    log_file = os.path.join(os.path.dirname(__file__), 'q1_miqcp.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def main():
    """主函数：执行球体模型计算"""
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("Q1 球体模型 - 开始")
    logging.info("=" * 60)
    
    cfg_path = os.path.join(PROJ_ROOT, "config.json")
    cfg = json.load(open(cfg_path, "r"))
    
    logging.info(f"配置参数:")
    logging.info(f"  投放时间: {cfg['t_release']}s")
    logging.info(f"  延迟时间: {cfg['delay_to_burst']}s")
    logging.info(f"  烟幕云半径: {cfg['radius']}m")
    logging.info(f"  有效时间: {cfg['effective_after_burst']}s")
    
    times, Ms, T, Cs, rs = build_series(cfg)
    
    total, intervals, t0, t1 = solve_scan(cfg)
    
    logging.info(f"球体模型结果:")
    logging.info(f"  总遮蔽时间: {total:.6f} 秒")
    logging.info(f"  遮蔽区间: {intervals}")
    logging.info(f"  时间窗口: [{t0:.6f}, {t1:.6f}]")
    
    print(f"球体模型: {total:.6f} 秒, 区间: {intervals}")
    
    logging.info("Q1 球体模型完成!")

if __name__ == "__main__":
    main()
