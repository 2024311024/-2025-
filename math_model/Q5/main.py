

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, linear_sum_assignment


# 真目标参数
TRUE_TARGET = {
    "r": 7,  # 圆柱半径
    "h": 10,  # 圆柱高度
    "center": np.array([0, 200, 0]),  # 底面圆心（注意：真目标不在原点）
    "sample_points": None  # 采样点
}

# 导弹参数（朝原点(0,0,0)的假目标飞行）
MISSILES = {
    "M1": {"init_pos": np.array([20000, 0, 2000]), "dir": None, "flight_time": None},
    "M2": {"init_pos": np.array([19000, 600, 2100]), "dir": None, "flight_time": None},
    "M3": {"init_pos": np.array([18000, -600, 1900]), "dir": None, "flight_time": None}
}
MISSILE_SPEED = 300  # 导弹速度(m/s)
G = 9.8  # 重力加速度(m/s²)
SMOKE_RADIUS = 10  # 烟幕有效半径(m)
SMOKE_SINK_SPEED = 3  # 起爆后下沉速度(m/s)
SMOKE_EFFECTIVE_TIME = 20  # 起爆后有效时长(s)

# 无人机参数
DRONES = {
    "FY1": {"init_pos": np.array([17800, 0, 1800]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY2": {"init_pos": np.array([12000, 1400, 1400]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY3": {"init_pos": np.array([6000, -3000, 700]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY4": {"init_pos": np.array([11000, 2000, 1800]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY5": {"init_pos": np.array([13000, -2000, 1300]), "max_smoke": 3, "speed_range": [70, 140],
            "smokes": [], "speed": None, "direction": None, "optimized": False}
}
DROP_INTERVAL = 1.0  # 同一无人机投放间隔(s)
TIME_STEP = 0.1  # 时间采样步长(s)

# 生成真目标采样点
def generate_true_target_samples():
    samples = []
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]
    # 底面采样：中心 + 圆周
    samples.append(center)
    for theta in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, center[2]]))
    # 顶面采样：中心 + 圆周
    top_center = center + np.array([0, 0, h])
    samples.append(top_center)
    for theta in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        x = top_center[0] + r * np.cos(theta)
        y = top_center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, top_center[2]]))
    # 侧面采样（加密层数与每层点数，可按算力调整）
    for z in np.linspace(center[2], top_center[2], 6):
        for theta in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            samples.append(np.array([x, y, z]))
    TRUE_TARGET["sample_points"] = np.array(samples)

# 初始化导弹（指向原点假目标）
def init_missiles():
    for m_name, m_data in MISSILES.items():
        init_pos = m_data["init_pos"]
        dir_vec = -init_pos / (np.linalg.norm(init_pos) + 1e-9)  # 指向 (0,0,0)
        m_data["dir"] = dir_vec * MISSILE_SPEED
        # 飞行到原点所需时间（上限）
        m_data["flight_time"] = np.linalg.norm(init_pos) / MISSILE_SPEED

generate_true_target_samples()
init_missiles()


def get_missile_pos(m_name, t):
    m_data = MISSILES[m_name]
    if t > m_data["flight_time"]:
        return m_data["init_pos"] + m_data["dir"] * m_data["flight_time"]
    return m_data["init_pos"] + m_data["dir"] * t

def get_drone_pos(drone_name, t):
    drone = DRONES[drone_name]
    if drone["speed"] is None or drone["direction"] is None:
        return drone["init_pos"]
    v_vec = np.array([drone["speed"] * np.cos(drone["direction"]),
                      drone["speed"] * np.sin(drone["direction"]), 0.0])
    return drone["init_pos"] + v_vec * t

def get_smoke_pos(drone_name, drop_time, det_delay, t):
    """返回 t 时刻烟幕中心位置；若未起爆或超寿命则返回 None。
       落地处理：若 det_z<0 则夹到 0.1（视为贴地仍有效，与你原逻辑一致）。"""
    drone = DRONES[drone_name]
    if t < drop_time:
        return None

    drop_pos = get_drone_pos(drone_name, drop_time)
    if t < drop_time + det_delay:
        # 投放后到起爆前：自由落体 + 随机械速度平移
        delta_t = t - drop_time
        v_vec = np.array([drone["speed"] * np.cos(drone["direction"]),
                          drone["speed"] * np.sin(drone["direction"]), 0.0])
        x = drop_pos[0] + v_vec[0] * delta_t
        y = drop_pos[1] + v_vec[1] * delta_t
        z = drop_pos[2] - 0.5 * G * delta_t ** 2
        return np.array([x, y, max(z, 0.1)])

    det_time = drop_time + det_delay
    if t > det_time + SMOKE_EFFECTIVE_TIME:
        return None

    # 起爆后：从起爆点以 3m/s 下沉
    delta_t_det = det_delay
    v_vec = np.array([drone["speed"] * np.cos(drone["direction"]),
                      drone["speed"] * np.sin(drone["direction"]), 0.0])
    det_x = drop_pos[0] + v_vec[0] * delta_t_det
    det_y = drop_pos[1] + v_vec[1] * delta_t_det
    det_z = drop_pos[2] - 0.5 * G * (delta_t_det ** 2)
    if det_z < 0:
        det_z = 0.1  # 和你原逻辑一致：贴地近似有效

    delta_t_after = t - det_time
    z = det_z - SMOKE_SINK_SPEED * delta_t_after
    return np.array([det_x, det_y, max(z, 0.1)])

def segment_sphere_intersect(p1, p2, center, radius):
    """线段 p1->p2 与球 (center, radius) 是否相交（最近点判定）"""
    vec_p = p2 - p1
    vec_c = center - p1
    denom = np.dot(vec_p, vec_p) + 1e-12
    t = np.dot(vec_c, vec_p) / denom
    if 0.0 <= t <= 1.0:
        nearest = p1 + t * vec_p
    else:
        nearest = p1 if t < 0.0 else p2
    return np.linalg.norm(nearest - center) <= radius + 1e-8


def active_clouds_at(t, all_smokes):
    """构造 t 时刻的活跃云团列表（每项为中心坐标 np.array([x,y,z])）"""
    clouds = []
    for s in all_smokes:
        pos = get_smoke_pos(s["drone"], s["drop_time"], s["det_delay"], t)
        if pos is not None:
            clouds.append(pos)
    return clouds

def is_target_blocked_by_any_clouds(m_pos, clouds):
    """允许多云团“分工”：对圆柱体每个采样点，只要被任一云团遮住即可。"""
    if not clouds:
        return False
    samples = TRUE_TARGET["sample_points"]
    for sample in samples:
        covered = False
        for c in clouds:
            if segment_sphere_intersect(m_pos, sample, c, SMOKE_RADIUS):
                covered = True
                break
        if not covered:
            return False
    return True

def evaluate_joint_block_time(all_smokes, t_step=TIME_STEP):
    """统计时间轴上“三枚导弹同时被遮蔽”的总时长（最终口径）。"""
    if not all_smokes:
        return 0.0
    t_min = 0.0
    t_max = max([MISSILES[m]["flight_time"] for m in MISSILES])
    joint = 0.0
    t = t_min
    while t <= t_max + 1e-9:
        clouds = active_clouds_at(t, all_smokes)
        if clouds:
            ok = True
            for m in MISSILES.keys():
                m_pos = get_missile_pos(m, t)
                if not is_target_blocked_by_any_clouds(m_pos, clouds):
                    ok = False
                    break
            if ok:
                joint += t_step
        t += t_step
    return joint

def calc_smoke_effective_time(drone_name, m_name, drop_time, det_delay):
    drone = DRONES[drone_name]
    v, theta = drone["speed"], drone["direction"]
    if v is None or theta is None:
        return -1000.0
    vmin, vmax = drone["speed_range"]
    if not (vmin - 1e-3 <= v <= vmax + 1e-3):
        return -1000.0

    # 起爆点有效性（近似）
    det_time = drop_time + det_delay
    drop_pos = get_drone_pos(drone_name, drop_time)
    det_z = drop_pos[2] - 0.5 * G * (det_delay ** 2)
    if det_z < -0.5:
        return -1000.0

    # 同机投放间隔
    for smoke in drone["smokes"]:
        if abs(drop_time - smoke["drop_time"]) < (DROP_INTERVAL - 1e-6):
            return -1000.0

    # 单弹×单导弹的“整圆柱被遮蔽”时长（参考性指标）
    max_t = min(det_time + SMOKE_EFFECTIVE_TIME, MISSILES[m_name]["flight_time"] + 1.0)
    min_t = max(det_time, 0.0)
    if min_t >= max_t - 1e-9:
        return 0.0

    eff = 0.0
    t = min_t
    while t <= max_t + 1e-9:
        m_pos = get_missile_pos(m_name, t)
        smoke_pos = get_smoke_pos(drone_name, drop_time, det_delay, t)
        if smoke_pos is not None:
            # 注意：这里仍按“单云团遮住全部采样点”作为启发式强约束
            all_intersect = True
            for sample in TRUE_TARGET["sample_points"]:
                if not segment_sphere_intersect(m_pos, sample, smoke_pos, SMOKE_RADIUS):
                    all_intersect = False
                    break
            if all_intersect:
                eff += TIME_STEP
        t += TIME_STEP
    return eff

def optimize_drone_trajectory(drone_name, m_name, retry=0):
    drone = DRONES[drone_name]
    v_min, v_max = drone["speed_range"]
    max_smoke = drone["max_smoke"]

    v_candidates = np.linspace(v_min, v_max, 8)
    best_v = None
    best_smokes = []
    max_total_time = 0.0

    for v in v_candidates:
        drone["speed"] = float(v)
        temp_smokes = []
        for i in range(max_smoke):
            min_drop_time = temp_smokes[-1]["drop_time"] + DROP_INTERVAL if temp_smokes else 0.0
            max_drop_time = MISSILES[m_name]["flight_time"] - 0.1
            if min_drop_time >= max_drop_time - 1e-9:
                break

            def objective(x):
                theta, drop_time, det_delay = x
                drone["direction"] = float(theta)
                return -calc_smoke_effective_time(drone_name, m_name, float(drop_time), float(det_delay))

            result = differential_evolution(
                func=objective,
                bounds=[(0, 2*np.pi), (min_drop_time, max_drop_time), (0.1, 10.0)],
                mutation=0.7,
                recombination=0.8,
                popsize=50,
                maxiter=60,
                disp=False
            )

            theta_opt, drop_time_opt, det_delay_opt = result.x
            drone["direction"] = float(theta_opt)
            effective_time = calc_smoke_effective_time(drone_name, m_name, float(drop_time_opt), float(det_delay_opt))
            if effective_time > 0.1:
                smoke = {
                    "v": float(v),
                    "theta": float(theta_opt),
                    "drop_time": float(drop_time_opt),
                    "det_delay": float(det_delay_opt),
                    "det_time": float(drop_time_opt + det_delay_opt),
                    "det_pos": get_smoke_pos(drone_name, float(drop_time_opt), float(det_delay_opt),
                                             float(drop_time_opt + det_delay_opt)),
                    "effective_time": float(effective_time),
                    "missile": m_name
                }
                temp_smokes.append(smoke)

        total_time = sum([s["effective_time"] for s in temp_smokes]) if temp_smokes else 0.0
        if total_time > max_total_time:
            max_total_time = total_time
            best_v = float(v)
            best_smokes = temp_smokes

    if not best_smokes and retry < 3:
        print(f"[{drone_name}] 优化失败，重试 {retry + 1}/3 ...")
        return optimize_drone_trajectory(drone_name, m_name, retry + 1)

    # 方向拟合 + 小波动精修（与原逻辑一致）
    if best_smokes:
        drop_points = []
        weights = []
        for smoke in best_smokes:
            v_vec = np.array([smoke["v"] * np.cos(smoke["theta"]), smoke["v"] * np.sin(smoke["theta"]), 0.0])
            drop_pos = drone["init_pos"] + v_vec * smoke["drop_time"]
            drop_points.append(drop_pos[:2])
            weights.append(smoke["effective_time"])
        drop_points = np.array(drop_points)
        weights = np.array(weights)

        X = np.column_stack([drop_points[:, 0], np.ones(len(drop_points))])
        W = np.diag(weights + 1e-9)
        try:
            k, b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ drop_points[:, 1])
            ref_theta = float(np.arctan(k))
        except np.linalg.LinAlgError:
            ref_theta = float(np.mean([s["theta"] for s in best_smokes])) if best_smokes else 0.0

        for i, smoke in enumerate(best_smokes):
            theta_candidates = [ref_theta - np.pi/24, ref_theta, ref_theta + np.pi/24]
            drop_candidates  = [smoke["drop_time"] - 0.8, smoke["drop_time"], smoke["drop_time"] + 0.8]

            best_effect = smoke["effective_time"]
            best_params = (smoke["theta"], smoke["drop_time"])

            for theta in theta_candidates:
                for drop_time in drop_candidates:
                    prev_drop = best_smokes[i-1]["drop_time"] if i > 0 else -1e9
                    if drop_time < prev_drop + (DROP_INTERVAL - 1e-6):
                        continue
                    drone["direction"] = float(theta)
                    effect = calc_smoke_effective_time(drone_name, m_name, float(drop_time), float(smoke["det_delay"]))
                    if effect > best_effect:
                        best_effect = effect
                        best_params = (float(theta), float(drop_time))

            smoke["theta"], smoke["drop_time"] = best_params
            smoke["det_time"] = float(smoke["drop_time"] + smoke["det_delay"])
            smoke["det_pos"]  = get_smoke_pos(drone_name, smoke["drop_time"], smoke["det_delay"], smoke["det_time"])
            smoke["effective_time"] = float(best_effect)

    drone["speed"] = best_v
    drone["direction"] = float(ref_theta) if best_smokes else None
    drone["smokes"] = best_smokes
    return best_smokes

# ============================ 4. 任务分配与迭代优化（新增 joint 统计打印） ============================
def assign_tasks(unoptimized_drones=None):
    if unoptimized_drones is None:
        unoptimized_drones = list(DRONES.keys())
    missile_list = list(MISSILES.keys())
    n_drones, n_missiles = len(unoptimized_drones), len(missile_list)
    if n_drones == 0:
        return {m: [] for m in missile_list}

    cost_matrix = np.zeros((n_drones, n_missiles))
    for i, d_name in enumerate(unoptimized_drones):
        d_init = DRONES[d_name]["init_pos"]
        d_avg_v = (DRONES[d_name]["speed_range"][0] + DRONES[d_name]["speed_range"][1]) / 2.0
        for j, m_name in enumerate(missile_list):
            m_init = MISSILES[m_name]["init_pos"]
            m_flight_time = MISSILES[m_name]["flight_time"]
            dist = np.linalg.norm(d_init - m_init)
            cost1 = dist / d_avg_v
            cost2 = 1000.0 / (m_flight_time + 1.0)
            cost3 = abs(d_avg_v - MISSILE_SPEED) / 100.0
            cost_matrix[i][j] = cost1 + cost2 + cost3

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = {m: [] for m in missile_list}
    for i, j in zip(row_ind, col_ind):
        assignments[missile_list[j]].append(unoptimized_drones[i])

    assigned_rows = set(row_ind)
    for i in range(n_drones):
        if i not in assigned_rows:
            min_cost_j = int(np.argmin(cost_matrix[i]))
            assignments[missile_list[min_cost_j]].append(unoptimized_drones[i])
    return assignments

def iterative_optimization(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3):
    # 重置无人机状态
    for d_name in DRONES:
        DRONES[d_name]["optimized"] = False
        DRONES[d_name]["smokes"] = []
        DRONES[d_name]["speed"] = None
        DRONES[d_name]["direction"] = None

    all_smokes = []
    prev_total_time = 0.0
    stall_count = 0

    for iteration in range(max_iterations):
        print(f"\n===== 迭代 {iteration + 1}/{max_iterations} =====")
        drones_without_solution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"尚未找到有效解的无人机: {drones_without_solution}")

        if not drones_without_solution:
            print("所有无人机都已找到启发式有效解，进入统计与可视化阶段。")
            break

        assignments = assign_tasks(drones_without_solution)

        current_total_time = 0.0
        iteration_smokes = []
        optimized_this_iter = []

        for m_name, drone_names in assignments.items():
            for d_name in drone_names:
                if len(DRONES[d_name]["smokes"]) > 0:
                    continue
                print(f"正在优化无人机 {d_name} 干扰 {m_name} ...")
                smokes = optimize_drone_trajectory(d_name, m_name)

                if smokes:
                    drone_smokes = [{**smoke, "drone": d_name} for smoke in smokes]
                    iteration_smokes.extend(drone_smokes)
                    current_total_time += sum([s["effective_time"] for s in smokes])
                    print(f"[{d_name}] 成功：{len(smokes)} 枚烟幕弹，单导弹累计口径 +{current_total_time:.2f}s")
                else:
                    print(f"[{d_name}] 本轮未找到有效投放方案（将继续尝试）")

                DRONES[d_name]["optimized"] = True
                optimized_this_iter.append(d_name)

        # 合并到全局方案
        all_smokes.extend(iteration_smokes)

        # —— 统计口径：单导弹累计（参考）与三弹同时（最终） ——
        total_single = sum([sum([s["effective_time"] for s in d["smokes"]]) for d in DRONES.values()])
        total_joint  = evaluate_joint_block_time(all_smokes, t_step=TIME_STEP)

        print(f"当前单导弹累计口径：{total_single:.2f}s")
        print(f"当前三弹同时遮蔽（最终口径）：{total_joint:.2f}s")
        print(f"本轮优化无人机: {optimized_this_iter}")
        print(f"已有有效解的无人机数量: {len([d for d in DRONES if len(DRONES[d]['smokes'])>0])}/{len(DRONES)}")

        improvement = total_single - prev_total_time
        print(f"相比上一轮（单导弹累计口径）改进量: {improvement:.2f}s")

        if improvement < improvement_threshold:
            stall_count += 1
            print(f"连续无显著改进次数: {stall_count}/{max_stall_iter}")
            if stall_count >= max_stall_iter:
                if drones_without_solution:
                    print(f"连续{max_stall_iter}轮无显著改进，但仍有无人机无解，继续尝试 ...")
                    stall_count = max_stall_iter - 1
                else:
                    print(f"连续{max_stall_iter}轮无显著改进，提前停止启发式阶段。")
                    break
        else:
            stall_count = 0

        prev_total_time = total_single

    # 结束检查
    remaining = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
    if remaining:
        print(f"\n警告：达到最大迭代次数，以下无人机仍未找到启发式有效解: {remaining}")

    return all_smokes


def save_result(smokes, filename="smoke_optimization_result.xlsx"):
    data = []
    for i, smoke in enumerate(smokes, 1):
        det_pos = smoke["det_pos"] if smoke["det_pos"] is not None else np.array([0, 0, 0])
        data.append({
            "烟幕弹编号": f"S{i}",
            "无人机编号": smoke.get("drone", "UNK"),
            "速度(m/s)": round(float(smoke["v"]), 2),
            "方向(°)": round(float(np.degrees(smoke["theta"])), 2),
            "投放时刻(s)": round(float(smoke["drop_time"]), 2),
            "起爆延迟(s)": round(float(smoke["det_delay"]), 2),
            "起爆时刻(s)": round(float(smoke["det_time"]), 2),
            "起爆点X(m)": round(float(det_pos[0]), 2),
            "起爆点Y(m)": round(float(det_pos[1]), 2),
            "起爆点Z(m)": round(float(det_pos[2]), 2),
            "干扰导弹": smoke["missile"],
            "（启发式）单弹遮蔽时长(s)": round(float(smoke["effective_time"]), 2)
        })
    df = pd.DataFrame(data)
    # 确保文件保存在Q5文件夹中
    output_path = os.path.join(os.path.dirname(__file__), filename)
    try:
        df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"结果已保存到 {output_path}")
    except Exception as e:
        print(f"[WARN] 写入 {output_path} 失败：{e}\n将改为保存 CSV 文件。")
        csv_name = output_path.replace(".xlsx", ".csv")
        df.to_csv(csv_name, index=False, encoding="utf-8-sig")
        print(f"结果已保存到 {csv_name}")
    return df


# ============================ 主函数 ============================
if __name__ == "__main__":
    all_smokes = iterative_optimization(max_iterations=20, improvement_threshold=0.3, max_stall_iter=3)

    # 汇总并输出
    if all_smokes:
        # 保存 Excel/CSV
        _ = save_result(all_smokes)

        # 两种口径的最终统计
        sum_single = sum([sum([s.get('effective_time', 0.0) for s in d["smokes"]]) for d in DRONES.values()])
        sum_joint  = evaluate_joint_block_time(all_smokes, t_step=TIME_STEP)

        print("\n" + "=" * 60)
        print("最终结果汇总：")
        print(f"（参考）单导弹累计口径总遮蔽时长：{sum_single:.2f} s")
        print(f"（最终）三弹同时遮蔽总时长 ：{sum_joint:.2f} s")

        print("\n各无人机投放详情：")
        for d_name, d_data in DRONES.items():
            if d_data["smokes"]:
                total = sum([s.get("effective_time", 0.0) for s in d_data["smokes"]])
                print(f"{d_name}：{len(d_data['smokes'])} 枚弹，单弹累计口径 {total:.2f} s")
            else:
                print(f"{d_name}：未找到启发式有效投放方案")
        print("=" * 60)
    else:
        print("未找到有效的烟幕弹投放方案（启发式阶段无解）。")
