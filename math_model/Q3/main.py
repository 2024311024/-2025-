
import os, sys, math, random, logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional



def setup_logger(log_path="q3_solver.log"):
    logger = logging.getLogger("Q3Solver")
    # 防止重复添加 handler（多次运行 import 时）
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 文件
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # 格式
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

log = setup_logger("q3_solver.log")


EPS = 1e-15

@dataclass
class ProblemConsts:
    # Target geometry
    true_target_bottom_center: Tuple[float, float, float] = (0, 200, 0)
    true_target_radius: float = 7.0
    true_target_height: float = 10.0
    fake_target_center: Tuple[float, float, float] = (0, 0, 0)

    # Missile
    M1_init: Tuple[float, float, float] = (20000, 0, 2000)
    missile_speed: float = 300.0

    # UAV
    FY1_init: Tuple[float, float, float] = (17800, 0, 1800)
    uav_speed_min: float = 70.0
    uav_speed_max: float = 140.0

    # Physics
    g: float = 9.8

    # Smoke cloud
    smoke_radius: float = 10.0
    smoke_sink_speed: float = 3.0
    smoke_effective_seconds: float = 20.0

    # Decision variable bounds
    t1_range: Tuple[float, float] = (0.5, 8.0)
    gap_range: Tuple[float, float] = (1.0, 8.0)
    delay_range: Tuple[float, float] = (2.0, 6.0)

    # Simulation
    t_end: float = 80.0
    dt: float = 0.02

    # Target sampling density
    side_n_theta: int = 48
    side_n_z: int = 20
    top_n_r: int = 8
    top_n_theta: int = 48

    # Random seed
    seed: int = 2025

@dataclass
class PSOConfig:
    num_particles: int = 60
    iters: int = 120
    w: float = 0.72
    c1: float = 1.49
    c2: float = 1.49
    vmax_frac: float = 0.2
    early_stopping_rounds: int = 20
    seed: int = 2025

# 决策变量边界: [heading_deg, speed, t1, gap12, gap23, d1, d2, d3]
PSO_BOUNDS = np.array([
    [0.0, 360.0],
    [70.0, 140.0],
    [0.5, 8.0],
    [1.0, 8.0],
    [1.0, 8.0],
    [2.0, 6.0],
    [2.0, 6.0],
    [2.0, 6.0],
], dtype=np.float64)

# ============================ 几何与工具 ============================
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS: return np.zeros_like(v)
    return v / n

def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)

def sample_dense_points_on_cylinder(center, radius, height,
                                    side_n_theta=48, side_n_z=20,
                                    top_n_r=8, top_n_theta=48) -> np.ndarray:
    cx, cy, cz = center
    pts = []

    theta = np.linspace(0, 2*np.pi, side_n_theta, endpoint=False)
    z_side = np.linspace(0, height, side_n_z)
    for t in theta:
        for z in z_side:
            x = cx + radius*np.cos(t)
            y = cy + radius*np.sin(t)
            pts.append([x, y, cz+z])

    r_top = np.linspace(0, radius, top_n_r)
    theta_top = np.linspace(0, 2*np.pi, top_n_theta, endpoint=False)
    for r in r_top:
        for t in theta_top:
            x = cx + r*np.cos(t)
            y = cy + r*np.sin(t)
            pts.append([x, y, cz+height])

    return np.array(pts, dtype=np.float64)

def heading_from_angle(angle_deg: float) -> np.ndarray:
    rad = np.radians(angle_deg)
    return np.array([np.cos(rad), np.sin(rad), 0.0])

def repair_decision_vector(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[0] = y[0] % 360.0
    if y[0] < 0: y[0] += 360.0
    for i in range(len(y)):
        y[i] = clamp(y[i], bounds[i,0], bounds[i,1])
    y[3] = max(y[3], 1.0)
    y[4] = max(y[4], 1.0)
    t3 = y[2] + y[3] + y[4]
    if t3 > 60.0:
        scale = 50.0 / max(t3, 1e-9)
        y[3] *= scale; y[4] *= scale
    return y

def decode_decision_vector(x: np.ndarray):
    heading_deg = x[0]; speed = x[1]
    t1, gap12, gap23 = x[2], x[3], x[4]
    d1, d2, d3 = x[5], x[6], x[7]
    t2 = t1 + gap12; t3 = t2 + gap23
    designs = [(t1, d1), (t2, d2), (t3, d3)]
    return heading_deg, speed, designs


class CylinderTarget:
    def __init__(self, center, radius, height,
                 side_n_theta=48, side_n_z=20, top_n_r=8, top_n_theta=48):
        self.center = np.array(center, dtype=np.float64)
        self.radius = radius
        self.height = height
        self.points = sample_dense_points_on_cylinder(
            center, radius, height, side_n_theta, side_n_z, top_n_r, top_n_theta
        )
        log.info(f"[Target] sampling points = {len(self.points)}")

class Missile:
    def __init__(self, init_pos, aim_pos, speed: float):
        self.init_pos = np.array(init_pos, dtype=np.float64)
        self.aim_pos  = np.array(aim_pos, dtype=np.float64)
        self.speed = speed
        self.velocity = speed * unit(self.aim_pos - self.init_pos)
    def pos(self, t: float) -> np.ndarray:
        return self.init_pos + self.velocity * t

class UAV:
    def __init__(self, init_pos, speed: float, heading_deg: float):
        self.init_pos = np.array(init_pos, dtype=np.float64)
        self.speed = speed
        self.heading_deg = heading_deg
        self.velocity = speed * heading_from_angle(heading_deg)
    def pos(self, t: float) -> np.ndarray:
        return self.init_pos + self.velocity * t

@dataclass
class BombDesign:
    t_release: float
    delay: float

class SmokeCloud:
    def __init__(self, uav: UAV, design: BombDesign, g: float,
                 sink_speed: float, effective_seconds: float):
        self.uav = uav
        self.design = design
        self.g = g
        self.sink_speed = sink_speed
        self.effective_seconds = effective_seconds

        self.t_explode = design.t_release + design.delay
        rel_pos = uav.pos(design.t_release)
        vx, vy = uav.velocity[0], uav.velocity[1]
        dz = -0.5 * g * design.delay**2
        self.explosion_pos = rel_pos + np.array([vx*design.delay, vy*design.delay, dz])
        self.t_effective_start = self.t_explode
        self.t_effective_end   = self.t_explode + effective_seconds

    def center(self, t: float) -> Optional[np.ndarray]:
        if t < self.t_effective_start or t > self.t_effective_end:
            return None
        sink = self.sink_speed * (t - self.t_explode)
        return self.explosion_pos + np.array([0.0, 0.0, -sink])

    def is_effective(self, t: float) -> bool:
        return self.t_effective_start <= t <= self.t_effective_end

# ============================ 遮蔽判定 ============================
class Scene:
    def __init__(self, consts: ProblemConsts):
        self.consts = consts
        self.target = CylinderTarget(
            consts.true_target_bottom_center,
            consts.true_target_radius,
            consts.true_target_height,
            consts.side_n_theta,
            consts.side_n_z,
            consts.top_n_r,
            consts.top_n_theta
        )
        self.missile = Missile(consts.M1_init, consts.fake_target_center, consts.missile_speed)

    def occluded(self, clouds: List[SmokeCloud], t_grid: np.ndarray) -> np.ndarray:
        K = len(self.target.points)
        T = len(t_grid)
        occluded = np.zeros(T, dtype=bool)
        R2 = self.consts.smoke_radius**2

        for i, t in enumerate(t_grid):
            M = self.missile.pos(t)

            # 收集当前有效云团中心
            eff_centers = []
            for c in clouds:
                p = c.center(t)
                if p is not None:
                    eff_centers.append(p)
            if not eff_centers:
                continue

            AB = self.target.points - M  # (K,3)
            denom = np.sum(AB*AB, axis=1)
            denom = np.clip(denom, EPS, None)

            # 多云团“协同”遮挡：任何一个云团遮住任意采样点都算本时刻有效
            for C_t in eff_centers:
                AQ = C_t - M
                num = np.dot(AB, AQ)          # (K,)
                t_star = np.clip(num/denom, 0.0, 1.0)
                Q_star = M + AB * t_star[:,None]
                d2 = np.sum((C_t - Q_star)**2, axis=1)
                if np.any(d2 <= R2):
                    occluded[i] = True
                    break
        return occluded

    def total_time(self, clouds: List[SmokeCloud]):
        t_grid = np.arange(0, self.consts.t_end + self.consts.dt, self.consts.dt)
        occ = self.occluded(clouds, t_grid)
        total = np.sum(occ) * self.consts.dt
        return total, t_grid, occ

    def fitness(self, uav: UAV, clouds: List[SmokeCloud]) -> float:
        try:
            total, _, _ = self.total_time(clouds)
            penalty = 1.0
            for c in clouds:
                if c.explosion_pos[2] < -100:
                    penalty *= 0.1
            for c in clouds:
                if c.design.t_release > 50:
                    penalty *= 0.5
            return total * penalty
        except Exception as e:
            log.error(f"[Fitness error] {e}", exc_info=True)
            return 0.0

# ============================ PSO 求解器 ============================
class PSOSolver:
    def __init__(self, consts: ProblemConsts, scene: Scene, config: PSOConfig):
        self.consts = consts
        self.scene = scene
        self.config = config
        np.random.seed(config.seed); random.seed(config.seed)

        self.dim = PSO_BOUNDS.shape[0]
        self.bounds = PSO_BOUNDS
        self.vmax = (self.bounds[:,1] - self.bounds[:,0]) * config.vmax_frac

        self.num_particles = config.num_particles
        self.particles = self._init_particles()
        self.vel = np.zeros((self.num_particles, self.dim))

        self.pbest_pos = self.particles.copy()
        self.pbest_val = np.zeros(self.num_particles)
        self._evaluate_particles()

        best_idx = np.argmax(self.pbest_val)
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.gbest_val = self.pbest_val[best_idx]
        self.last_improve_iter = 0
        self.best_history = [self.gbest_val]

    def _init_particles(self):
        P = np.zeros((self.num_particles, self.dim))
        for i in range(self.num_particles):
            for j in range(self.dim):
                lo, hi = self.bounds[j]
                P[i,j] = random.uniform(lo, hi)
        return P

    def _fitness(self, x: np.ndarray) -> float:
        try:
            xr = repair_decision_vector(x, self.bounds)
            heading_deg, speed, designs = decode_decision_vector(xr)
            uav = UAV(self.consts.FY1_init, speed, heading_deg)
            clouds = [SmokeCloud(uav, BombDesign(t_release, delay),
                                 self.consts.g, self.consts.smoke_sink_speed,
                                 self.consts.smoke_effective_seconds)
                      for (t_release, delay) in designs]
            return self.scene.fitness(uav, clouds)
        except Exception as e:
            log.error(f"[Eval error] {e}", exc_info=True)
            return 0.0

    def _evaluate_particles(self):
        for i in range(self.num_particles):
            self.pbest_val[i] = self._fitness(self.particles[i])

    def _update_velocity(self, i: int):
        r1, r2 = random.random(), random.random()
        cog = self.config.c1 * r1 * (self.pbest_pos[i] - self.particles[i])
        soc = self.config.c2 * r2 * (self.gbest_pos - self.particles[i])
        self.vel[i] = self.config.w * self.vel[i] + cog + soc
        self.vel[i] = np.clip(self.vel[i], -self.vmax, self.vmax)

    def _update_position(self, i: int):
        self.particles[i] += self.vel[i]
        self.particles[i] = repair_decision_vector(self.particles[i], self.bounds)

    def step(self, it: int):
        for i in range(self.num_particles):
            self._update_velocity(i)
            self._update_position(i)
            fit = self._fitness(self.particles[i])
            if fit > self.pbest_val[i]:
                self.pbest_val[i] = fit
                self.pbest_pos[i] = self.particles[i].copy()
                if fit > self.gbest_val:
                    self.gbest_val = fit
                    self.gbest_pos = self.particles[i].copy()
                    self.last_improve_iter = it
        self.best_history.append(self.gbest_val)

    def solve(self):
        log.info(f"[PSO] particles={self.num_particles}, iters={self.config.iters}")
        log.info(f"[PSO] init best = {self.gbest_val:.6f}")
        for it in range(1, self.config.iters+1):
            self.step(it)
            if it % 10 == 0 or it == 1:
                log.info(f"[PSO] iter {it:3d} | best = {self.gbest_val:.6f}")
            if (it - self.last_improve_iter) > self.config.early_stopping_rounds:
                log.info(f"[PSO] early stop at iter {it} (no improve for {self.config.early_stopping_rounds})")
                break
        log.info(f"[PSO] done. final best = {self.gbest_val:.6f}")
        return self.gbest_pos, self.gbest_val

# ============================ 写表（result1.xlsx） ============================
class ResultWriter:
    COLS = [
        "无人机运动方向",
        "无人机运动速度 (m/s)",
        "烟幕干扰弹编号",
        "烟幕干扰弹投放点的x坐标 (m)",
        "烟幕干扰弹投放点的y坐标 (m)",
        "烟幕干扰弹投放点的z坐标 (m)",
        "烟幕干扰弹起爆点的x坐标 (m)",
        "烟幕干扰弹起爆点的y坐标 (m)",
        "烟幕干扰弹起爆点的z坐标 (m)",
        "有效干扰时长 (s)"
    ]
    NOTE = "注：以x轴为正向，逆时针方向为正，取值0~360（度）。"

    def write(self, heading_deg: float, speed: float, designs: List[Tuple[float, float]],
              uav: UAV, clouds: List[SmokeCloud], total_time: float, path: str = "result1.xlsx") -> None:
        rows = []
        for i, (t_release, delay) in enumerate(designs, start=1):
            p_rel = uav.pos(t_release)
            p_exp = clouds[i-1].explosion_pos
            rows.append({
                "无人机运动方向": round(float(heading_deg), 3),
                "无人机运动速度 (m/s)": round(float(speed), 3),
                "烟幕干扰弹编号": i,
                "烟幕干扰弹投放点的x坐标 (m)": round(float(p_rel[0]), 3),
                "烟幕干扰弹投放点的y坐标 (m)": round(float(p_rel[1]), 3),
                "烟幕干扰弹投放点的z坐标 (m)": round(float(p_rel[2]), 3),
                "烟幕干扰弹起爆点的x坐标 (m)": round(float(p_exp[0]), 3),
                "烟幕干扰弹起爆点的y坐标 (m)": round(float(p_exp[1]), 3),
                "烟幕干扰弹起爆点的z坐标 (m)": round(float(p_exp[2]), 3),
                "有效干扰时长 (s)": np.nan
            })

        rows.append({c: np.nan for c in self.COLS})
        note_row = {c: np.nan for c in self.COLS}
        note_row["无人机运动方向"] = self.NOTE
        rows.append(note_row)

        df = pd.DataFrame(rows, columns=self.COLS)
        df.iloc[0, df.columns.get_loc("有效干扰时长 (s)")] = round(float(total_time), 6)
        try:
            df.to_excel(path, index=False, engine="openpyxl")
            log.info(f"[Excel] saved -> {path}")
        except Exception as e:
            alt = path.replace(".xlsx", ".csv")
            df.to_csv(alt, index=False, encoding="utf-8-sig")
            log.warning(f"[Excel] openpyxl 写入失败：{e}，已改存 CSV：{alt}")

# ============================ 主程序 ============================
def main():
    log.info("="*64)
    log.info("Q3: PSO Optimization (single-file, with logging) -> result1.xlsx")
    log.info("="*64)

    consts = ProblemConsts()
    cfg = PSOConfig()

    log.info(f"[Const] target center={consts.true_target_bottom_center}, r={consts.true_target_radius}, h={consts.true_target_height}")
    log.info(f"[Const] missile init={consts.M1_init}, v={consts.missile_speed}")
    log.info(f"[Const] uav init={consts.FY1_init}, v in [{consts.uav_speed_min}, {consts.uav_speed_max}]")
    log.info(f"[Const] smoke R={consts.smoke_radius}, sink={consts.smoke_sink_speed}, life={consts.smoke_effective_seconds}")
    log.info(f"[Sim ] T={consts.t_end}, dt={consts.dt}")

    try:
        scene = Scene(consts)
        solver = PSOSolver(consts, scene, cfg)
        x_best, f_best = solver.solve()

        heading_deg, speed, designs = decode_decision_vector(x_best)
        log.info("[Best Solution]")
        log.info(f"  heading = {heading_deg:.3f} deg")
        log.info(f"  speed   = {speed:.2f} m/s")
        for i,(tr, dl) in enumerate(designs, 1):
            log.info(f"  bomb{i}: release={tr:.3f}s, delay={dl:.3f}s")

        uav = UAV(consts.FY1_init, speed, heading_deg)
        clouds = [SmokeCloud(uav, BombDesign(tr, dl), consts.g, consts.smoke_sink_speed, consts.smoke_effective_seconds)
                  for (tr, dl) in designs]

        total_time, t_grid, occluded = scene.total_time(clouds)
        log.info(f"[Detail] total occlusion time = {total_time:.6f} s")
        log.info(f"[Detail] grid: {len(t_grid)} steps, from {t_grid[0]:.3f} to {t_grid[-1]:.3f} s")
        log.info(f"[Detail] occluded steps: {np.sum(occluded)} / {len(occluded)}")

        # 写表
        writer = ResultWriter()
        out_path = os.path.join(os.path.dirname(__file__) or ".", "result1.xlsx")
        writer.write(heading_deg, speed, designs, uav, clouds, total_time, out_path)


        log.info("="*64)
        log.info("Q3 Optimization Complete! Output: result1.xlsx")
        log.info("="*64)

    except Exception as e:
        log.error(f"[Fatal] {e}", exc_info=True)

if __name__ == "__main__":
    main()
