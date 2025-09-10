
from __future__ import annotations
import argparse, csv, json, math, os, random, sys, time, logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

from Q1 import method_A_strict as mas  # 提供 solve_scan_strict(cfg)


# === 搜索空间默认范围 ===
ANGLE_MIN, ANGLE_MAX = 0.0, 360.0
SPEED_MIN, SPEED_MAX = 70.0, 140.0
TR_MIN, TR_MAX       = 0.5, 3.0
DL_MIN, DL_MAX       = 2.0, 5.0

# === 载入基础配置 ===
with open(os.path.join(PROJ_ROOT, "config.json"), "r", encoding="utf-8") as f:
    BASE_CFG: Dict[str, Any] = json.load(f)


@dataclass
class GAParams:
    pop_size: int = 40
    generations: int = 80
    elite_frac: float = 0.1
    tourney_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    sigma_angle: float = 2.0
    sigma_speed: float = 8.0
    sigma_t_release: float = 0.25
    sigma_delay: float = 0.25
    delta_t_coarse: float = 0.02
    delta_t_fine: float = 0.01
    seed: int | None = 42


@dataclass
class Individual:
    angle_deg: float
    speed: float
    t_release: float
    delay: float
    fitness: float = float("-inf")

    def genome(self) -> List[float]:
        return [self.angle_deg, self.speed, self.t_release, self.delay]

    def copy(self) -> "Individual":
        return Individual(*self.genome(), fitness=self.fitness)


# === 工具：角度 wrap、边界钳制 ===
def wrap_angle_deg(angle: float) -> float:
    a = angle % 360.0
    if a < 0:
        a += 360.0
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def heading_from_beta(beta_rad: float) -> List[float]:
    return [math.cos(beta_rad), math.sin(beta_rad), 0.0]


# === 适应度评估（目标：最大化严格遮蔽时长） ===
def evaluate(ind: Individual, delta_t: float, bounds: Dict[str, Tuple[float, float]]) -> float:
    cfg = json.loads(json.dumps(BASE_CFG))  # 深拷贝
    beta = math.radians(ind.angle_deg)
    cfg["uav"]["heading"] = heading_from_beta(beta)
    cfg["uav"]["speed"] = float(clamp(ind.speed, *bounds["speed"]))
    cfg["t_release"] = float(clamp(ind.t_release, *bounds["t_release"]))
    cfg["delay_to_burst"] = float(clamp(ind.delay, *bounds["delay"]))
    cfg["delta_t"] = float(delta_t)
    try:
        total, _, _, _ = mas.solve_scan_strict(cfg)
    except Exception:
        total = 0.0
    return float(total)


# === 初始化 ===
def random_individual(bounds: Dict[str, Tuple[float, float]]) -> Individual:
    return Individual(
        angle_deg = random.uniform(*bounds["angle_deg"]),
        speed     = random.uniform(*bounds["speed"]),
        t_release = random.uniform(*bounds["t_release"]),
        delay     = random.uniform(*bounds["delay"]),
    )


def init_population(n: int, params: GAParams, bounds: Dict[str, Tuple[float, float]]) -> List[Individual]:
    pop = [random_individual(bounds) for _ in range(n)]
    for ind in pop:
        ind.fitness = evaluate(ind, params.delta_t_coarse, bounds)
    return pop


# === 选择（锦标赛） ===
def tournament_select(pop: List[Individual], k: int) -> Individual:
    cand = random.sample(pop, k)
    return max(cand, key=lambda x: x.fitness)


# === 交叉（简化线性混合；可替换为 SBX） ===
def crossover(p1: Individual, p2: Individual, rate: float, bounds: Dict[str, Tuple[float, float]]) -> Tuple[Individual, Individual]:
    if random.random() > rate:
        return p1.copy(), p2.copy()

    def blend(a: float, b: float, low: float, high: float) -> float:
        t = random.random()
        x = a * t + b * (1 - t)
        return clamp(x, low, high)

    c1 = Individual(
        angle_deg = wrap_angle_deg(blend(p1.angle_deg, p2.angle_deg, *bounds["angle_deg"])),
        speed     = blend(p1.speed,     p2.speed,     *bounds["speed"]),
        t_release = blend(p1.t_release, p2.t_release, *bounds["t_release"]),
        delay     = blend(p1.delay,     p2.delay,     *bounds["delay"]),
    )
    c2 = Individual(
        angle_deg = wrap_angle_deg(blend(p2.angle_deg, p1.angle_deg, *bounds["angle_deg"])),
        speed     = blend(p2.speed,     p1.speed,     *bounds["speed"]),
        t_release = blend(p2.t_release, p1.t_release, *bounds["t_release"]),
        delay     = blend(p2.delay,     p1.delay,     *bounds["delay"]),
    )
    return c1, c2


# === 变异（高斯扰动 + 边界处理 + 角度 wrap） ===
def mutate(ind: Individual, params: GAParams, bounds: Dict[str, Tuple[float, float]]) -> None:
    if random.random() > params.mutation_rate:
        return
    ind.angle_deg = wrap_angle_deg(ind.angle_deg + random.gauss(0.0, params.sigma_angle))
    ind.speed     = clamp(ind.speed     + random.gauss(0.0, params.sigma_speed),     *bounds["speed"])
    ind.t_release = clamp(ind.t_release + random.gauss(0.0, params.sigma_t_release), *bounds["t_release"])
    ind.delay     = clamp(ind.delay     + random.gauss(0.0, params.sigma_delay),     *bounds["delay"])


# === 代际更新（精英保留 + 锦标赛 + 交叉 + 变异 + 评估） ===
def evolve(pop: List[Individual], params: GAParams, bounds: Dict[str, Tuple[float, float]]) -> List[Individual]:
    pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
    elite_n = max(1, int(len(pop) * params.elite_frac))
    next_pop: List[Individual] = [pop[i].copy() for i in range(elite_n)]  # 精英直接保留

    while len(next_pop) < len(pop):
        p1 = tournament_select(pop, params.tourney_size)
        p2 = tournament_select(pop, params.tourney_size)
        c1, c2 = crossover(p1, p2, params.crossover_rate, bounds)
        mutate(c1, params, bounds)
        c1.fitness = evaluate(c1, params.delta_t_coarse, bounds)
        next_pop.append(c1)
        if len(next_pop) < len(pop):
            mutate(c2, params, bounds)
            c2.fitness = evaluate(c2, params.delta_t_coarse, bounds)
            next_pop.append(c2)
    return next_pop


# === 运行 GA 并保存结果（包含 intervals 与 t_window） ===
def run_ga(params: GAParams) -> Dict[str, Any]:
    # 设置日志
    log_file = os.path.join(os.path.dirname(__file__), 'q2_ga.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 60)
    logging.info("Q2 遗传算法 - 开始")
    logging.info("=" * 60)
    
    if params.seed is not None:
        random.seed(params.seed)

    bounds = {
        "angle_deg": (ANGLE_MIN, ANGLE_MAX),
        "speed": (SPEED_MIN, SPEED_MAX),
        "t_release": (TR_MIN, TR_MAX),
        "delay": (DL_MIN, DL_MAX),
    }

    pop = init_population(params.pop_size, params, bounds)
    best = max(pop, key=lambda x: x.fitness).copy()
    t0 = time.time()

    for gen in range(params.generations):
        pop = evolve(pop, params, bounds)
        curr_best = max(pop, key=lambda x: x.fitness)
        if curr_best.fitness > best.fitness:
            best = curr_best.copy()
        print(f"[Gen {gen:03d}] best={best.fitness:.6f}  angle={best.angle_deg:.3f}  v={best.speed:.2f}  tr={best.t_release:.2f}  dl={best.delay:.2f}")

    # 末尾精细复核，用更小 delta_t，并保存区间与时间窗
    cfg_best = json.loads(json.dumps(BASE_CFG))
    cfg_best["uav"]["heading"] = heading_from_beta(math.radians(best.angle_deg))
    cfg_best["uav"]["speed"] = float(best.speed)
    cfg_best["t_release"] = float(best.t_release)
    cfg_best["delay_to_burst"] = float(best.delay)
    cfg_best["delta_t"] = float(params.delta_t_fine)
    total_fine, intervals, t0_win, t1_win = mas.solve_scan_strict(cfg_best)

    # 输出最终结果到终端
    print("\n" + "=" * 60)
    print("Q2 Genetic Algorithm Optimization Complete!")
    print("=" * 60)
    print(f"Best solution found:")
    print(f"  Angle (degrees): {best.angle_deg:.6f}")
    print(f"  Speed (m/s): {best.speed:.6f}")
    print(f"  Release time (s): {best.t_release:.6f}")
    print(f"  Delay (s): {best.delay:.6f}")
    print(f"  Total occlusion time: {total_fine:.6f} seconds")
    print(f"  Intervals: {intervals}")
    print(f"  Time window: [{t0_win:.6f}, {t1_win:.6f}]")
    print(f"  Runtime: {time.time() - t0:.3f} seconds")
    print("=" * 60)
    
    logging.info("Q2 遗传算法完成!")

    return {
        "best_individual": best,
        "total_fine": total_fine,
        "intervals": intervals,
        "t_window": [t0_win, t1_win],
        "runtime_sec": time.time() - t0
    }




def parse_args() -> GAParams:
    ap = argparse.ArgumentParser(description="Q2 Genetic Algorithm (strict occlusion)")
    ap.add_argument("--pop-size", type=int, default=40)
    ap.add_argument("--generations", type=int, default=80)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--tourney-size", type=int, default=3)
    ap.add_argument("--crossover-rate", type=float, default=0.9)
    ap.add_argument("--mutation-rate", type=float, default=0.3)
    ap.add_argument("--sigma-angle", type=float, default=2.0)
    ap.add_argument("--sigma-speed", type=float, default=8.0)
    ap.add_argument("--sigma-t-release", type=float, default=0.25)
    ap.add_argument("--sigma-delay", type=float, default=0.25)
    ap.add_argument("--delta-t-coarse", type=float, default=0.02)
    ap.add_argument("--delta-t-fine", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    return GAParams(
        pop_size=args.pop_size,
        generations=args.generations,
        elite_frac=args.elite_frac,
        tourney_size=args.tourney_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        sigma_angle=args.sigma_angle,
        sigma_speed=args.sigma_speed,
        sigma_t_release=args.sigma_t_release,
        sigma_delay=args.sigma_delay,
        delta_t_coarse=args.delta_t_coarse,
        delta_t_fine=args.delta_t_fine,
        seed=args.seed,
    )


def main() -> None:
    params = parse_args()
    run_ga(params)


if __name__ == "__main__":
    main()



