"""world.py —— 自动消防小车仿真内核

一台带机械臂的消防小车，在连续坐标的城市路网中完成：
- 火情检测 → 立刻自检（结果供前端可视化）→ 出发灭火；
- 水量低于 10% 自动前往最近补水站；灭火时允许用到 0% 再补水；
- 功能故障自动前往最近维修站；
- 行驶中遇到路障，停车用机械臂清障；
- 遇到行人自动降速慢行，行人通过后恢复；
- 行人随机生成，走人行道、过斑马线并遵守红绿灯；
- 路网常驻四辆私家车：路口让行、靠边避让、同侧排队等消防车起步，任何时候都不重叠。

所有状态由后端统一维护，update(dt) 是唯一推进仿真时间的入口，便于测试。
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from algo.fov import (
    fov_astar, line_of_sight, point_in_fov, visibility_polygon, visible_nodes,
)
from algo.pathfinding import graph_astar
from algo.route_planner import plan as plan_multi_route
from sim.city import CityMap

BASE_SPEED = 95.0        # 正常巡航速度（单位/秒）
SLOW_SPEED = 26.0        # 遇到行人时的慢行速度
ACCEL = 120.0            # 加速度
TURN_RATE = 2.8          # 最大转向角速度（弧度/秒）
LANE = CityMap.LANE_OFFSET

WATER_LOW = 10.0         # 水量低阈值
REFILL_RATE = 18.0       # 补水速度（%/秒）
SPRAY_WATER = 7.0        # 灭火耗水（%/秒）
SPRAY_POWER = 24.0       # 灭火强度下降（点/秒）
REPAIR_TIME = 4.0
SELF_CHECK_STEP = 0.38   # 每个自检项目耗时
AI_SCAN_TIME = 1.4       # AI 巡检点在点停留时间（秒）
AI_ARRIVE = 42.0         # 判定「到达停靠点」的距离

# ---- FOV（视野）感知寻路：车头前方 110°、300 单位的视野锥 ----
FOV_HALF_ANGLE = math.radians(55.0)
FOV_RADIUS = 300.0
FOV_RAYS = 41
FOV_INVISIBLE_PENALTY = 1.28   # 视野外的路网边增加探索代价
FOV_ARM_RANGE = 132.0          # 这个距离内的正前方路障交给机械臂，不抢方向
FOV_OBSTACLE_ROAD_DIST = 135.0 # 路障离最近道路节点不超过这个距离才视为占道
FOV_RANGE_BY_KIND = {
    "fire": 300.0,
    "obstacle": 300.0,
    "pedestrian": 220.0,
    "car": 260.0,
}

# ---- FOV 感知反馈：把视野里的异常整理成结构化事件，交给 AI 重新规划 ----
FOV_HAZARD_TTL = 8.0        # 行人 / 拥堵这类会消失的临时封锁保持时间（秒）
FOV_FEEDBACK_CD = 6.0       # 同一对象 / 节点重复反馈的最小间隔
FOV_PED_NOTICE_DIST = 150.0 # 行人进入这个距离才向 AI 反馈「占道」
FOV_JAM_CARS = 2            # 视野内同一路口停着几辆车判定为拥堵
AI_REPLAN_CD = 2.0          # AI 重新规划的最小间隔，避免每帧重排

# 车道横向半宽：行人只有真正走在车道里才会把消防车逼停，
# 人行道上的行人（距车道中心 ~31 单位）不再触发停车。
PED_LANE_HALF = 24.0
PED_BRAKE_DIST = 52.0
HEAVY_BRAKE_DIST = 72.0     # 不可移开路障进入车头这个距离就停车，等待重规划

# ---- 私家车（NPC）：常驻路网，两辆，给消防车让行 ----
CAR_MAX = 2              # 路网常驻私家车数量（不再随机刷新/消失）
CAR_SPEED_MIN = 52.0     # 私家车巡航速度区间（比消防车慢，便于被追上时让行）
CAR_SPEED_MAX = 68.0
CAR_ACCEL = 90.0
CAR_BRAKE = 160.0
CAR_TURN = 3.2
CAR_SPAWN_CLEAR = 90.0   # 出生点离消防车的最小距离
CAR_ROUTE_MIN = 320.0    # 优先给私家车派一条够长的路线，避免刚上路就到终点
CAR_YIELD_DIST = 150.0   # 消防车进入这个范围，私家车开始让行
CAR_CLEAR_DIST = 190.0   # 消防车退出这个范围，让行解除（带回滞，避免抖动）
CAR_INTERSECTION_DIST = 78.0   # 离路口多近算「在路口」
CAR_PULL_OVER = 32.0     # 让行时向路肩平移的距离（消防车很宽，要让到路肩才不重叠）
CAR_WAIT_GAP = 52.0      # 同侧等消防车时，停在它后面的距离
CAR_TRUCK_LEN = 30.0     # 与消防车的纵向安全距离（消防车已缩到和小车一样大）
CAR_TRUCK_SIDE = 20.0    # 与消防车的横向安全距离（并排时的车宽）
CAR_CAR_LEN = 32.0       # 两辆小车之间的纵向安全距离
CAR_CAR_SIDE = 24.0      # 两辆小车之间的横向安全距离

# ---- 路口拥堵疏通：先散人 → 小车绕行 → 消防车清理 ----
JAM_R = 100.0            # 判定「堵在这个路口」的范围
JAM_TRIGGER = 3.0        # 连续堵多久判定为拥堵
JAM_PED_TIME = 1.8       # 疏散行人的时间
JAM_CAR_TIME = 2.2       # 小车重新找路的时间
JAM_PED_R = 115.0        # 需要疏散的行人范围
JAM_TRUCK_DWELL = 2.0    # 消防车到场清理停留时间
JAM_PULL_TIME = 6.0      # 拥堵时小车靠边让行的时长
JAM_DEMO_TIME = 18.0     # 演示用临时拥堵车辆停住的时间

# ---- 消防车被堵 / 故障：先让小车撤离，消防车再回去维修 ----
CLEARWAY_R = 130.0       # 需要撤离的小车范围
CLEARWAY_TRIGGER = 3.0   # 消防车被堵多久判定为堵住
CLEARWAY_TIME = 2.5      # 小车撤离的时间
ESCAPE_TIME = 8.0        # 小车撤离时忽略消防车的时长
CAR_SIDE_RATE = 26.0     # 横向平移速度

# 车身颜色：与消防车同款的深红，只做细微深浅区分
CAR_COLORS = ("#c8102e", "#a80f27", "#d7263d", "#9e0b1f")

FAULT_NAMES = {
    "arm": "机械臂",
    "pump": "灭火水泵",
    "sensor": "传感器",
    "nav": "导航定位",
}

SELF_CHECK_ITEMS = [
    ("water", "水量与水箱"),
    ("battery", "电池电量"),
    ("arm", "机械臂"),
    ("pump", "灭火水泵"),
    ("sensor", "传感器阵列"),
    ("nav", "导航与定位"),
]


@dataclass
class Truck:
    x: float
    y: float
    heading: float
    speed: float = 0.0
    water: float = 100.0
    battery: float = 100.0
    u_turn: bool = False     # 正在原地掉头
    goal_node: int | None = None   # 当前路线的目标节点，供重新规划用
    state: str = "idle"
    route: list[int] = field(default_factory=list)
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    wp_index: int = 0
    target_fire: int | None = None
    target_station: int | None = None
    slowing: bool = False
    fault: str | None = None
    arm_target: int | None = None
    arm_progress: float = 0.0
    spray: bool = False


@dataclass
class Fire:
    id: int
    x: float
    y: float
    anchor_node: int
    intensity: float = 100.0
    checked: bool = False
    unreachable: bool = False
    custom: bool = False       # 由地图编辑器放置（用于「清空编辑内容」）


@dataclass
class Obstacle:
    id: int
    x: float
    y: float
    custom: bool = False
    # barrier：普通路障，机械臂可以移开；heavy：水泥墩 / 施工围挡，移不开，只能绕行
    kind: str = "barrier"

    @property
    def immovable(self) -> bool:
        return self.kind == "heavy"


@dataclass
class Pedestrian:
    id: int
    x: float
    y: float
    speed: float
    emoji: str
    path: list[int] = field(default_factory=list)
    idx: int = 1
    waiting: bool = False
    wait_kind: str = ""
    state: str = "walking"
    custom: bool = False       # 由地图编辑器放置


@dataclass
class NpcCar:
    """路网上的私家车：只负责交通，不参与消防任务，遇消防车要让行。"""

    id: int
    x: float
    y: float
    heading: float
    cruise: float
    color: str
    route: list[int] = field(default_factory=list)
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    wp_index: int = 0
    speed: float = 0.0
    state: str = "driving"          # driving / yield_intersection / yield_roadside
    side: float = 0.0               # 路侧避让时向路缘平移的距离（只影响绘制位置）
    target_node: int | None = None
    retry: float = 0.0              # 到达终点后重选路线的冷却
    stuck: float = 0.0              # 被堵住的计时
    reverse: float = 0.0            # 倒车让行的剩余时间
    escape: float = 0.0             # 消防车被堵/故障时的撤离计时
    pull_aside: float = 0.0         # 拥堵时靠边让行的剩余时间
    hold: float = 0.0               # >0 时原地停住（用于演示路口拥堵）
    temporary: bool = False         # 拥堵演示用的临时车辆，到点自动消失

    @property
    def yielding(self) -> bool:
        return self.state != "driving"


class World:
    def __init__(self, auto_spawn: bool = True, seed: int | None = None):
        self.rng = random.Random(seed)
        self.city = CityMap()
        self.auto_spawn = auto_spawn
        self.time = 0.0
        self.truck = Truck(x=154.0, y=300.0, heading=math.pi / 2)
        self.fires: list[Fire] = []
        self.obstacles: list[Obstacle] = []
        self.pedestrians: list[Pedestrian] = []
        self.cars: list[NpcCar] = []
        self.events: list[dict] = []
        self.stats = {
            "fires_out": 0,
            "refills": 0,
            "repairs": 0,
            "obstacles_cleared": 0,
            "self_checks": 0,
            "ai_routes": 0,
            "ai_stops": 0,
            "jam_clears": 0,
        }
        self.ai_mission: dict | None = None
        self.jam: dict | None = None
        self.clear_way: dict | None = None
        self._truck_stuck = 0.0
        self._ped_block_timer = 0.0
        self._heavy_block_timer = 0.0
        self._heavy_block_at = -99.0

        # FOV：视野锥 + 可见节点 / 实体 + 已探索节点
        self.fov_nav = True
        self._fov_seen_nodes: set[int] = set()
        self._fov_visible_nodes: set[int] = set()
        self._fov_known_obstacles: dict[int, int] = {}
        self._fov_polygon: list[tuple[float, float]] = []
        self._fov_replans = 0
        self._fov_last_replan: float | None = None
        # FOV 感知反馈：临时危险（行人占道 / 拥堵）+ 结构化反馈事件
        self._fov_hazards: dict[str, dict] = {}
        self._fov_feedback: list[dict] = []
        self._fov_feedback_at: dict[str, float] = {}
        self._fov_counts = {"obstacle": 0, "pedestrian": 0, "jam": 0}
        self._ai_replan_at = -99.0
        self.fov = self._empty_fov()

        self._jam_watch: dict[int, float] = {}
        self._jam_clear_timer = 0.0
        self._next_fire_id = 1
        self._next_obstacle_id = 1
        self._next_ped_id = 1
        self._next_car_id = 1
        self._check_index = 0
        self._check_timer = 0.0
        self._repair_timer = 0.0
        self._fire_timer = 6.0
        self._obstacle_timer = 18.0
        self._fault_timer = 45.0
        self._car_timer = 2.0        # 第一辆私家车稍早出现，方便演示
        self.self_check = self._empty_self_check()
        self._log("系统启动，消防小车在待命点完成初始化", "info")
        for _ in range(10):
            self._spawn_pedestrian()
        if auto_spawn:
            for _ in range(CAR_MAX):
                self.spawn_car()
        self._update_fov(0.0)

    # ------------------------------------------------------------------ 基础工具
    @staticmethod
    def _empty_fov() -> dict:
        return {
            "enabled": True,
            "half_angle_deg": round(math.degrees(FOV_HALF_ANGLE), 1),
            "radius": FOV_RADIUS,
            "heading": 0.0,
            "polygon": [],
            "visible_nodes": [],
            "seen_nodes": 0,
            "entities": [],
            "blocked_nodes": [],
            "hard_nodes": [],
            "hard_marks": [],
            "hazards": [],
            "feedback": [],
            "counts": {"obstacle": 0, "pedestrian": 0, "jam": 0},
            "path": [],
            "replans": 0,
            "last_replan": None,
        }

    @staticmethod
    def _empty_self_check() -> dict:
        return {
            "running": False,
            "reason": None,
            "fire_id": None,
            "progress": 0.0,
            "passed": None,
            "started_at": None,
            "finished_at": None,
            "steps": [
                {"key": key, "name": name, "status": "pending", "detail": "等待检测"}
                for key, name in SELF_CHECK_ITEMS
            ],
        }

    def _log(self, text: str, level: str = "info") -> None:
        self.events.append({"time": round(self.time, 1), "text": text, "level": level})
        if len(self.events) > 80:
            self.events = self.events[-60:]

    @property
    def signal(self) -> str:
        """南北绿灯 / 东西绿灯，20 秒一个完整周期。"""
        return "NS" if self.time % 20 < 10 else "EW"

    @property
    def signal_remaining(self) -> float:
        return round(10 - (self.time % 10), 1)

    def _dist(self, x1, y1, x2, y2) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    # ------------------------------------------------------------------ 主循环
    def update(self, dt: float) -> None:
        dt = max(0.0, min(float(dt), 0.2))
        if dt == 0:
            return
        self.time += dt
        self._update_pedestrians(dt)
        self._update_self_check(dt)
        self._update_fov(dt)
        self._update_truck(dt)
        self._update_cars(dt)
        self._update_clear_way(dt)
        self._update_ped_block(dt)
        self._update_heavy_block(dt)
        self._update_jam(dt)
        if self.auto_spawn:
            self._update_spawners(dt)

    def _update_spawners(self, dt: float) -> None:
        self._fire_timer -= dt
        if self._fire_timer <= 0:
            self.spawn_fire()
            self._fire_timer = self.rng.uniform(22, 34)
        self._obstacle_timer -= dt
        if self._obstacle_timer <= 0:
            heavy_count = len([o for o in self.obstacles if o.immovable])
            # 偶尔生成「移不开」的水泥墩，让 FOV 绕行在演示中自然出现
            kind = "heavy" if (heavy_count < 2 and self.rng.random() < 0.35) else "barrier"
            limit = 5 if kind == "heavy" else 4
            if len(self.obstacles) < limit:
                traveling = self.truck.state in (
                    "to_fire", "to_refill", "to_repair", "ai_mission")
                self.spawn_obstacle(ahead=traveling and self.rng.random() < 0.6, kind=kind)
            self._obstacle_timer = self.rng.uniform(16, 28)
        self._fault_timer -= dt
        if self._fault_timer <= 0:
            if self.truck.fault is None:
                self.inject_fault()
            self._fault_timer = self.rng.uniform(40, 65)
        self._car_timer -= dt
        if self._car_timer <= 0:
            # 常驻：正常情况下四辆一直在路上，只有数量不足时才补一辆
            if len(self.cars) < CAR_MAX:
                self.spawn_car()
            self._car_timer = 3.0

    # ------------------------------------------------------------------ 车辆逻辑
    def _update_truck(self, dt: float) -> None:
        t = self.truck
        if t.state == "self_check":
            t.speed = 0.0
            return
        if t.state in ("to_fire", "to_refill", "to_repair", "ai_mission"):
            if t.state == "to_fire" and not self._fire_by_id(t.target_fire):
                self._enter_idle("目标火情已消失")
                return
            if t.state == "ai_mission":
                self._update_arm(dt)
                self._drive(dt, self._target_speed())
                self._check_ai_arrival()
                return
            self._update_arm(dt)
            target_speed = self._target_speed()
            self._drive(dt, target_speed)
            self._check_arrival()
            return
        t.speed = 0.0
        if t.state == "extinguishing":
            self._extinguish(dt)
        elif t.state == "refilling":
            self._refill(dt)
        elif t.state == "repairing":
            self._repair(dt)
        elif t.state == "ai_scan":
            self._ai_scan(dt)
        elif t.state == "idle":
            self._idle()

    def _drive(self, dt: float, target_speed: float) -> None:
        t = self.truck
        if not t.waypoints or t.wp_index >= len(t.waypoints):
            target_speed = 0.0

        # 接近目标点时转向；大角度转弯先减速，避免“漂移”
        if t.waypoints and t.wp_index < len(t.waypoints):
            tx, ty = t.waypoints[t.wp_index]
            dx, dy = tx - t.x, ty - t.y
            dist = math.hypot(dx, dy)
            if dist < 14:
                t.wp_index += 1
                if t.wp_index >= len(t.waypoints):
                    target_speed = 0.0
            elif dist > 0:
                desired = math.atan2(dy, dx)
                diff = (desired - t.heading + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) > 2.0 or (t.u_turn and abs(diff) > 0.25):
                    # 目标在车后：原地转过头再走，不绕大圈、也不倒车
                    t.u_turn = True
                    t.speed = max(0.0, t.speed - ACCEL * 2.5 * dt)
                    if t.speed < 2.0:
                        t.heading += max(-TURN_RATE * 1.5 * dt,
                                         min(TURN_RATE * 1.5 * dt, diff))
                        t.heading = (t.heading + math.pi) % (2 * math.pi) - math.pi
                    return
                t.u_turn = False
                if abs(diff) > 0.95:
                    target_speed = min(target_speed, 30.0)
                if t.speed > 3:
                    t.heading += max(-TURN_RATE * dt, min(TURN_RATE * dt, diff))

        if t.speed < target_speed:
            t.speed = min(target_speed, t.speed + ACCEL * dt)
        else:
            t.speed = max(target_speed, t.speed - ACCEL * 1.6 * dt)

        nx = t.x + math.cos(t.heading) * t.speed * dt
        ny = t.y + math.sin(t.heading) * t.speed * dt
        nx = max(10.0, min(self.city.WIDTH - 10.0, nx))
        ny = max(10.0, min(self.city.HEIGHT - 10.0, ny))
        if self._truck_would_crowd(nx, ny):
            t.speed = 0.0                         # 前面真有车：停下来，不硬挤
        else:
            t.x, t.y = nx, ny
        t.heading = (t.heading + math.pi) % (2 * math.pi) - math.pi

    def _truck_would_crowd(self, nx: float, ny: float) -> bool:
        """消防车这一步会不会挤到小车：会就停住等它让开。"""
        for car in self.cars:
            cx, cy = self._car_drawn_xy(car)
            if not self._truck_blocked_at(nx, ny, cx, cy):
                continue
            if math.hypot(nx - cx, ny - cy) < math.hypot(self.truck.x - cx, self.truck.y - cy):
                return True
        return False

    def _truck_blocked_at(self, x: float, y: float, cx: float, cy: float) -> bool:
        """消防车放在 (x, y) 时，是否会压到 (cx, cy) 这辆小车。"""
        dx, dy = cx - x, cy - y
        c, s = math.cos(self.truck.heading), math.sin(self.truck.heading)
        return abs(dx * c + dy * s) < CAR_TRUCK_LEN and abs(-dx * s + dy * c) < CAR_TRUCK_SIDE

    def _target_speed(self) -> float:
        t = self.truck
        target = BASE_SPEED
        t.slowing = False
        if t.arm_target is not None:
            return 0.0
        # 只对「真正走进车道」的行人减速：人行道上的行人不该把车逼停
        cos_h, sin_h = math.cos(t.heading), math.sin(t.heading)
        for p in self.pedestrians:
            dx, dy = p.x - t.x, p.y - t.y
            dist = math.hypot(dx, dy)
            if dist > 130 or dist < 0.1:
                continue
            forward = dx * cos_h + dy * sin_h
            lateral = -dx * sin_h + dy * cos_h
            if forward <= 0 or abs(lateral) > PED_LANE_HALF:
                continue
            t.slowing = True
            target = min(target, 0.0 if dist < PED_BRAKE_DIST else SLOW_SPEED)
        if self._heavy_ahead():
            t.slowing = True
            target = 0.0
        return target

    def _heavy_ahead(self, max_fwd: float = HEAVY_BRAKE_DIST) -> Obstacle | None:
        """车头正前方、移不开的路障；撞上去也不会消失，只能绕开或等待。"""
        t = self.truck
        cos_h, sin_h = math.cos(t.heading), math.sin(t.heading)
        for obstacle in self.obstacles:
            if not obstacle.immovable:
                continue
            dx, dy = obstacle.x - t.x, obstacle.y - t.y
            forward = dx * cos_h + dy * sin_h
            lateral = -dx * sin_h + dy * cos_h
            if 0.0 < forward <= max_fwd and abs(lateral) <= 26.0:
                return obstacle
        return None

    def _update_heavy_block(self, dt: float) -> None:
        """被不可移开的路障顶住：停下来，让 FOV 反馈给 AI 重新规划。"""
        t = self.truck
        if not self._truck_busy() or t.speed > 1.0:
            self._heavy_block_timer = 0.0
            return
        obstacle = self._heavy_ahead()
        if obstacle is None:
            self._heavy_block_timer = 0.0
            return
        self._heavy_block_timer += dt
        if self._heavy_block_timer < 2.0:
            return
        self._heavy_block_timer = 0.0
        if self.ai_mission:
            self._ai_replan(f"不可移开路障 #{obstacle.id} 挡路")
        elif not self._replan_truck():
            if self.time - self._heavy_block_at > 6.0:
                self._heavy_block_at = self.time
                self._log(f"不可移开路障 #{obstacle.id} 挡住去路，等待人工处理", "critical")

    def _check_arrival(self) -> None:
        t = self.truck
        if t.state == "to_fire":
            fire = self._fire_by_id(t.target_fire)
            if fire is None:
                self._enter_idle("目标火情已消失")
                return
            ax, ay = self.city.road.xy[fire.anchor_node]
            if self._dist(t.x, t.y, ax, ay) <= 40:
                t.state = "extinguishing"
                t.spray = True
                t.speed = 0.0
                self._log(f"到达火场 #{fire.id}，开始喷水灭火", "info")
        elif t.state == "to_refill":
            st = self.city.stations[t.target_station] if t.target_station is not None else None
            if st:
                ax, ay = self.city.road.xy[st["node"]]
                if self._dist(t.x, t.y, ax, ay) <= 45:
                    t.state = "refilling"
                    t.speed = 0.0
                    self._log(f"抵达{st['name']}，开始补水", "info")
        elif t.state == "to_repair":
            st = self.city.stations[t.target_station] if t.target_station is not None else None
            if st:
                ax, ay = self.city.road.xy[st["node"]]
                if self._dist(t.x, t.y, ax, ay) <= 45:
                    t.state = "repairing"
                    t.speed = 0.0
                    self._repair_timer = 0.0
                    self._log(f"抵达{st['name']}，开始检修", "warning")

    def _enter_idle(self, reason: str = "") -> None:
        t = self.truck
        t.state = "idle"
        t.speed = 0.0
        t.spray = False
        t.route = []
        t.waypoints = []
        t.wp_index = 0
        t.target_fire = None
        t.target_station = None
        if reason:
            self._log(reason, "info")

    def _idle(self) -> None:
        t = self.truck
        if t.fault:
            if not self.self_check["running"]:
                self.start_self_check("fault")
            return
        if t.water < WATER_LOW:
            if self.dispatch_refill():
                return
        unchecked = [f for f in self.fires if not f.checked and not f.unreachable]
        if unchecked:
            self.start_self_check("fire", unchecked[0].id)
            return
        checked = [f for f in self.fires if f.checked and not f.unreachable]
        if checked:
            self.dispatch_fire(checked[0])

    def _extinguish(self, dt: float) -> None:
        t = self.truck
        fire = self._fire_by_id(t.target_fire)
        if fire is None:
            t.spray = False
            if self.ai_mission:
                self._advance_ai("目标火情已消失，继续 AI 路线")
            else:
                self._enter_idle()
            return
        if self._dist(t.x, t.y, fire.x, fire.y) > 100:
            t.spray = False
            if self.ai_mission:
                self._ai_goto(self.ai_mission["index"])     # 沿 AI 路线重新贴近火点
            else:
                t.state = "to_fire"
                self._plan_to_fire(fire)
            return
        t.spray = True
        t.water = max(0.0, t.water - SPRAY_WATER * dt)
        fire.intensity = max(0.0, fire.intensity - SPRAY_POWER * dt)
        if t.water <= 0:
            t.spray = False
            self._log("灭火剂耗尽，火情未完全扑灭，立即前往最近补水站", "critical")
            self._end_ai_mission("水量耗尽，AI 任务中断")
            self.dispatch_refill(resume_fire=fire.id)
            return
        if fire.intensity <= 0:
            self.fires = [f for f in self.fires if f.id != fire.id]
            self.stats["fires_out"] += 1
            t.spray = False
            self._log(f"火情 #{fire.id} 已扑灭（剩余水量 {t.water:.0f}%）", "success")
            if self.ai_mission:
                self._advance_ai(f"火情 #{fire.id} 已扑灭，AI 路线继续")
                return
            if t.water < WATER_LOW:
                self.dispatch_refill()
            else:
                self._enter_idle()

    def _refill(self, dt: float) -> None:
        t = self.truck
        t.water = min(100.0, t.water + REFILL_RATE * dt)
        t.battery = min(100.0, t.battery + 20.0 * dt)
        if t.water >= 100.0:
            t.water = 100.0
            self.stats["refills"] += 1
            self._log("补水完成，水量 100%", "success")
            self._enter_idle()

    def _repair(self, dt: float) -> None:
        self._repair_timer += dt
        if self._repair_timer >= REPAIR_TIME:
            t = self.truck
            old = t.fault
            t.fault = None
            self.stats["repairs"] += 1
            self._log(f"维修完成，{FAULT_NAMES.get(old or '', '故障')}已恢复正常", "success")
            self._enter_idle()

    # ------------------------------------------------------------------ 机械臂 / 障碍物
    def _update_arm(self, dt: float) -> None:
        t = self.truck
        if t.arm_target is not None:
            obstacle = next((o for o in self.obstacles if o.id == t.arm_target), None)
            if obstacle is None:
                t.arm_target = None
                t.arm_progress = 0.0
                return
            t.arm_progress = min(1.0, t.arm_progress + dt / 1.6)
            if t.arm_progress >= 1.0:
                self.obstacles = [o for o in self.obstacles if o.id != obstacle.id]
                self.stats["obstacles_cleared"] += 1
                self._log(f"机械臂已将路障 #{obstacle.id} 移出车道", "success")
                t.arm_target = None
                t.arm_progress = 0.0
            return
        nearest = None
        nearest_d = 9999.0
        for o in self.obstacles:
            if o.immovable:
                continue                     # 水泥墩 / 围挡清不掉，只能绕行
            dx, dy = o.x - t.x, o.y - t.y
            dist = math.hypot(dx, dy)
            if dist > 95 or dist < 0.1:
                continue
            diff = (math.atan2(dy, dx) - t.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < 0.5 and dist < nearest_d:
                nearest, nearest_d = o, dist
        if nearest is not None:
            t.arm_target = nearest.id
            t.arm_progress = 0.0
            self._log(f"前方发现路障 #{nearest.id}，停车使用机械臂清理", "warning")

    def _update_obstacles(self) -> None:
        # 障碍物本身是静态的；保留方法方便后续扩展（例如被火引燃）
        return

    # ------------------------------------------------------------------ FOV 感知
    def set_fov(self, enabled: bool | None = None) -> dict:
        """开启 / 关闭 FOV 感知寻路；不传参数时切换当前状态。"""
        self.fov_nav = (not self.fov_nav) if enabled is None else bool(enabled)
        self._update_fov(0.0)
        state = "开启" if self.fov_nav else "关闭"
        self._log(f"FOV 感知寻路已{state}", "info")
        return {"ok": True, "enabled": self.fov_nav, "msg": f"FOV 感知寻路已{state}"}

    def _fov_blocked_nodes(self) -> set[int]:
        """软封锁节点：可移开的路障 + 行人 / 拥堵等临时危险，可以绕但没有硬性断开。"""
        blocked: set[int] = set()
        alive = {o.id for o in self.obstacles}
        for oid in list(self._fov_known_obstacles):
            if oid not in alive:
                self._fov_known_obstacles.pop(oid, None)
                continue
            obstacle = next((o for o in self.obstacles if o.id == oid), None)
            if obstacle is None:
                continue
            # 近处正前方的路障交给机械臂清理，避免 FOV 让车绕开而永远清不到。
            if math.hypot(obstacle.x - self.truck.x,
                          obstacle.y - self.truck.y) < FOV_ARM_RANGE:
                continue
            blocked.add(self._fov_known_obstacles[oid])
        blocked |= self._fov_hazard_nodes()
        blocked.discard(self.truck.goal_node)
        return blocked

    def _heavy_blocked_nodes(self) -> set[int]:
        """硬封锁节点：不可移开的路障所在道路节点，任何路线都不许穿过。"""
        nodes: set[int] = set()
        for obstacle in self.obstacles:
            if not obstacle.immovable:
                continue
            node = self.city.nearest_road_node(obstacle.x, obstacle.y)
            nx, ny = self.city.road.xy[node]
            if math.hypot(obstacle.x - nx, obstacle.y - ny) <= FOV_OBSTACLE_ROAD_DIST:
                nodes.add(node)
        return nodes

    def _fov_hazard_nodes(self) -> set[int]:
        """行人占道 / 路口拥堵等临时危险占据的道路节点（带过期时间）。"""
        return {hz["node"] for hz in self._fov_hazards.values() if hz["node"] is not None}

    def _ped_in_lane(self, ped: Pedestrian, max_dist: float = FOV_PED_NOTICE_DIST) -> bool:
        """行人是否真的走在车道里：车头方向 + 横向不超过半条车道。"""
        t = self.truck
        dx, dy = ped.x - t.x, ped.y - t.y
        forward = dx * math.cos(t.heading) + dy * math.sin(t.heading)
        lateral = -dx * math.sin(t.heading) + dy * math.cos(t.heading)
        return 0.0 < forward <= max_dist and abs(lateral) <= PED_LANE_HALF

    def _set_hazard(self, key: str, kind: str, node: int | None,
                    x: float, y: float, text: str, ttl: float) -> None:
        self._fov_hazards[key] = {
            "key": key, "kind": kind, "node": node,
            "x": round(x, 1), "y": round(y, 1), "text": text,
            "until": round(self.time + ttl, 1),
        }

    def _push_feedback(self, kind: str, key: str, node: int | None,
                       x: float, y: float, text: str, action: str,
                       level: str = "warning") -> bool:
        """记录一条 FOV 感知反馈；同一对象有冷却时间，避免刷屏。"""
        if self.time - self._fov_feedback_at.get(key, -99.0) < FOV_FEEDBACK_CD:
            return False
        self._fov_feedback_at[key] = self.time
        self._fov_counts[kind] = self._fov_counts.get(kind, 0) + 1
        self._fov_feedback.insert(0, {
            "time": round(self.time, 1), "kind": kind, "key": key, "node": node,
            "x": round(x, 1), "y": round(y, 1),
            "text": text, "action": action, "level": level,
        })
        del self._fov_feedback[8:]
        self._log(f"{text}；{action}", level)
        return True

    def _fov_jammed_nodes(self) -> list[int]:
        """视野内出现「同一个路口停着 >= N 辆车」的拥堵节点。"""
        t = self.truck
        jammed: list[int] = []
        for node, (x, y) in enumerate(self.city.road.xy):
            if self._dist(x, y, t.x, t.y) > 240:
                continue
            if not self._fov_has_sight(x, y, "car"):
                continue
            stopped = 0
            for car in self.cars:
                cx, cy = self._car_drawn_xy(car)
                if car.speed < 5.0 and math.hypot(cx - x, cy - y) < 70.0:
                    stopped += 1
            if stopped >= FOV_JAM_CARS:
                jammed.append(node)
        return jammed

    def _scan_fov_hazards(self, entities: list[dict]) -> None:
        """把视野里的三类异常整理成反馈事件：不可移开路障 / 行人占道 / 路口拥堵。"""
        t = self.truck
        for key, hazard in list(self._fov_hazards.items()):
            if self.time >= hazard["until"]:
                self._fov_hazards.pop(key, None)

        seen_obstacles = {e["id"] for e in entities if e["kind"] == "obstacle"}
        # 1) 不可移开路障：机械臂清不掉，只能绕行
        for obstacle in self.obstacles:
            if obstacle.id not in seen_obstacles or not obstacle.immovable:
                continue
            node = self.city.nearest_road_node(obstacle.x, obstacle.y)
            key = f"obstacle:{obstacle.id}"
            self._set_hazard(key, "obstacle", node, obstacle.x, obstacle.y,
                             f"不可移开路障 #{obstacle.id}", FOV_HAZARD_TTL * 2)
            if self._push_feedback(
                    "obstacle", key, node, obstacle.x, obstacle.y,
                    f"FOV 发现不可移开路障 #{obstacle.id}，车道被封锁",
                    "标记节点，AI 重新规划绕行", "critical"):
                self._ai_feedback(node, f"不可移开路障 #{obstacle.id}")

        # 2) 行人占道：只统计站在车道里的行人，人行道上的不算
        for ped in self.pedestrians:
            if not self._ped_in_lane(ped):
                continue
            node = self.city.nearest_road_node(ped.x, ped.y)
            key = f"pedestrian:{ped.id}"
            dist = math.hypot(ped.x - t.x, ped.y - t.y)
            self._set_hazard(key, "pedestrian", node, ped.x, ped.y,
                             f"行人 #{ped.id} 占道", 4.0)
            if self._push_feedback(
                    "pedestrian", key, node, ped.x, ped.y,
                    f"FOV 识别到行人 #{ped.id} 占用车道（{dist:.0f} 单位）",
                    "减速避让，必要时停车等待", "warning"):
                # 只是路过就减速，只有真被行人卡住时才值得让 AI 重排路线
                if t.speed < 5.0:
                    self._ai_feedback(node, f"行人 #{ped.id} 占道")

        # 3) 路口拥堵：既有的拥堵状态优先，其次看视野里停住的车
        jam_node = self.jam["node"] if self.jam else None
        if jam_node is None:
            candidates = self._fov_jammed_nodes()
            jam_node = candidates[0] if candidates else None
        if jam_node is not None:
            jx, jy = self.city.road.xy[jam_node]
            key = f"jam:{jam_node}"
            self._set_hazard(key, "jam", jam_node, jx, jy, "路口拥堵", FOV_HAZARD_TTL)
            if self._push_feedback(
                    "jam", key, jam_node, jx, jy,
                    f"FOV 发现路口 #{jam_node} 拥堵（车辆停车等待）",
                    "标记节点，AI 重新规划绕行", "warning"):
                self._ai_feedback(jam_node, f"路口 #{jam_node} 拥堵")

    def _ai_feedback(self, node: int | None, reason: str) -> None:
        """把 FOV 的发现回传给 AI：命中当前剩余路线时重新规划。"""
        mission = self.ai_mission
        if not mission or not self.fov_nav or node is None:
            return
        t = self.truck
        remaining = set(t.route[t.wp_index:]) if t.route else set()
        if node not in remaining and node != t.goal_node:
            return
        self._ai_replan(reason)

    def _fov_has_sight(self, x: float, y: float, kind: str,
                       origin: tuple[float, float] | None = None) -> bool:
        t = self.truck
        ox, oy = origin or (t.x, t.y)
        radius = min(FOV_RADIUS, FOV_RANGE_BY_KIND.get(kind, FOV_RADIUS))
        if not point_in_fov((ox, oy), t.heading, FOV_HALF_ANGLE, radius, (x, y)):
            return False
        return line_of_sight((ox, oy), (x, y), self.city.buildings, margin=1.0)

    def _update_fov(self, dt: float = 0.0) -> None:
        """刷新视野锥、可见实体、已探索节点和感知反馈；发现危险时动态重规划。"""
        t = self.truck
        origin = (t.x, t.y)
        buildings = self.city.buildings

        self._fov_polygon = visibility_polygon(
            origin, t.heading, FOV_HALF_ANGLE, FOV_RADIUS, buildings, FOV_RAYS,
        )
        self._fov_visible_nodes = visible_nodes(
            self.city.road.xy, origin, t.heading, FOV_HALF_ANGLE,
            FOV_RADIUS, buildings,
        )
        self._fov_seen_nodes |= self._fov_visible_nodes

        entities: list[dict] = []

        def add(kind: str, eid: int, x: float, y: float) -> None:
            if not self._fov_has_sight(x, y, kind, origin):
                return
            entities.append({
                "kind": kind, "id": eid,
                "x": round(x, 1), "y": round(y, 1),
                "distance": round(math.hypot(x - t.x, y - t.y), 1),
            })

        for fire in self.fires:
            add("fire", fire.id, fire.x, fire.y)
        for obstacle in self.obstacles:
            add("obstacle", obstacle.id, obstacle.x, obstacle.y)
        for ped in self.pedestrians:
            add("pedestrian", ped.id, ped.x, ped.y)
        for car in self.cars:
            cx, cy = self._car_drawn_xy(car)
            add("car", car.id, cx, cy)

        # 只记住真正落在道路附近的障碍物，避免把路边装饰当成车道封锁。
        alive = {o.id: o for o in self.obstacles}
        for oid in list(self._fov_known_obstacles):
            if oid not in alive:
                self._fov_known_obstacles.pop(oid, None)
        newly_seen: set[int] = set()
        for ent in entities:
            if ent["kind"] != "obstacle":
                continue
            oid = ent["id"]
            if oid in self._fov_known_obstacles:
                continue
            obstacle = alive.get(oid)
            if obstacle is None:
                continue
            node = self.city.nearest_road_node(obstacle.x, obstacle.y)
            nx, ny = self.city.road.xy[node]
            if math.hypot(obstacle.x - nx, obstacle.y - ny) <= FOV_OBSTACLE_ROAD_DIST:
                self._fov_known_obstacles[oid] = node
                newly_seen.add(oid)

        # 三类异常 -> 结构化反馈；命中 AI 剩余路线时直接触发重新规划
        if self.fov_nav:
            self._scan_fov_hazards(entities)
        else:
            self._fov_hazards.clear()

        soft = self._fov_blocked_nodes()
        hard = self._heavy_blocked_nodes()
        blocked = sorted(soft | hard)
        path: list[list[float]] = []
        if self.fov_nav and t.state in (
                "to_fire", "to_refill", "to_repair", "ai_mission"):
            remaining = t.waypoints[t.wp_index:]
            if remaining:
                path = [[round(t.x, 1), round(t.y, 1)]]
                path += [[round(x, 1), round(y, 1)] for x, y in remaining]

        self.fov = {
            "enabled": self.fov_nav,
            "half_angle_deg": round(math.degrees(FOV_HALF_ANGLE), 1),
            "radius": FOV_RADIUS,
            "heading": round(t.heading, 3),
            "polygon": [[x, y] for x, y in self._fov_polygon],
            "visible_nodes": sorted(self._fov_visible_nodes),
            "seen_nodes": len(self._fov_seen_nodes),
            "entities": entities,
            "blocked_nodes": blocked,
            "hard_nodes": sorted(hard),
            "hard_marks": [
                {"node": n, "x": round(self.city.road.xy[n][0], 1),
                 "y": round(self.city.road.xy[n][1], 1)}
                for n in sorted(hard)
            ],
            "hazards": sorted(self._fov_hazards.values(),
                              key=lambda hz: hz["until"]),
            "feedback": list(self._fov_feedback),
            "counts": dict(self._fov_counts),
            "path": path,
            "replans": self._fov_replans,
            "last_replan": self._fov_last_replan,
        }

        # 新看到的路障 / 新出现的危险如果正压在当前剩余路线上，就地重新规划；
        # 没有绕行空间时保留原路线，交给机械臂 / 等待处理，绝不让车辆卡死。
        if not self.fov_nav or not self._truck_busy():
            return
        if not newly_seen and not self._fov_hazards:
            return
        remaining_route = set(t.route[t.wp_index:]) if t.route else set()
        hit = [n for oid, n in self._fov_known_obstacles.items()
               if oid in newly_seen and n in remaining_route and n != t.goal_node]
        if not hit:
            return
        reason = f"FOV 发现前方路障（节点 {hit[0]}）"
        if self.ai_mission:
            self._ai_replan(reason)
            return
        if self._replan_truck():
            self._fov_replans += 1
            self._fov_last_replan = round(self.time, 1)
            self.fov["replans"] = self._fov_replans
            self.fov["last_replan"] = self._fov_last_replan
            self._log(f"{reason}，已按可见道路重新规划绕行路线", "warning")

    # ------------------------------------------------------------------ 寻路 / 调度
    def _lane_waypoints(self, route: list[int]) -> list[tuple[float, float]]:
        """把道路节点序列转成「靠右车道」的连续航点。

        每个路口只保留「这一段的车道终点」，车到路口后直接朝下一段的车道拐弯，
        不额外绕到路口角上，所以不会出现兜圈子的动作。
        """
        pts = [self.city.road.xy[n] for n in route]
        waypoints: list[tuple[float, float]] = []
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 0.5:
                continue
            rx, ry = -dy / length, dx / length       # 行驶方向的右侧法向量
            waypoints.append((x2 + rx * LANE, y2 + ry * LANE))
        return waypoints

    def _nav_adj(self, blocked: set[int] | None = None) -> dict:
        """按封锁节点裁剪后的路网邻接表；给 AI 规划和 FOV 寻路共用。

        道路节点很少，直接复制一份比在 A* 里逐条判断更直观。
        """
        blocked = set(blocked if blocked is not None else self._heavy_blocked_nodes())
        if not blocked:
            return self.city.road.adj
        return {
            node: [(nxt, w) for nxt, w in edges if nxt not in blocked]
            for node, edges in self.city.road.adj.items()
            if node not in blocked
        }

    def _nearest_free_node(self, x: float, y: float,
                           blocked: set[int] | None = None) -> int:
        """离 (x,y) 最近的、没有被重障封锁的道路节点。"""
        blocked = blocked or set()
        if not blocked:
            return self.city.nearest_road_node(x, y)
        best, best_d = None, float("inf")
        for node, (nx, ny) in enumerate(self.city.road.xy):
            if node in blocked:
                continue
            d = math.hypot(nx - x, ny - y)
            if d < best_d:
                best, best_d = node, d
        if best is None:
            return self.city.nearest_road_node(x, y)
        return best

    def _plan_route(self, goal_node: int) -> tuple[list[int], float] | None:
        # 起点取「车头前方」的节点：避免掉头横穿马路，保持靠右行驶
        ahead = (self.truck.x + math.cos(self.truck.heading) * 45.0,
                 self.truck.y + math.sin(self.truck.heading) * 45.0)
        soft = self._fov_blocked_nodes() if self.fov_nav else set()
        hard = self._heavy_blocked_nodes()
        if goal_node in hard:
            return None                     # 目标点被不可移开的障碍物压住，无法停靠
        start_node = self._nearest_free_node(*ahead, hard)

        def try_fov(start: int, blocked: set[int]) -> list[int] | None:
            return fov_astar(
                self.city.road.adj, start, goal_node, self.city.road.xy,
                visible=self._fov_visible_nodes, blocked=blocked,
                invisible_penalty=FOV_INVISIBLE_PENALTY,
            )

        route = try_fov(start_node, soft | hard) if (soft or hard) else None
        if not route and soft:
            # 绕不过可移开的路障时允许穿过，交给机械臂清理（保持原有行为）
            route = try_fov(start_node, hard)
        if not route and not hard:
            # 平时仍走原有全局 A*，保证旧功能的行为不变
            route = graph_astar(self.city.road.adj, start_node, goal_node,
                                self.city.road.xy)
        if not route:
            start_node = self._nearest_free_node(self.truck.x, self.truck.y, hard)
            route = try_fov(start_node, soft | hard) if (soft or hard) else None
            if not route and soft:
                route = try_fov(start_node, hard)
            if not route and not hard:
                route = graph_astar(self.city.road.adj, start_node, goal_node,
                                    self.city.road.xy)
        if not route:
            return None
        length = 0.0
        for a, b in zip(route, route[1:]):
            (x1, y1), (x2, y2) = self.city.road.xy[a], self.city.road.xy[b]
            length += math.hypot(x2 - x1, y2 - y1)
        return route, length

    def _set_route(self, route: list[int], node: int) -> None:
        t = self.truck
        t.goal_node = node
        t.route = list(route)
        t.waypoints = self._lane_waypoints(route)
        entry = self._lane_entry(route)
        if entry is not None:
            t.waypoints.insert(0, entry)      # 先并回右侧车道，再沿车道走
        t.wp_index = 0
        # 注意：这里不清空 target_fire / target_station，
        # 否则 FOV 重规划会让 to_fire 误判「目标火情已消失」而放弃任务。
        # 终点节点就是目标停靠点；若航点为空，车辆已在目标点
        if not t.waypoints and node is not None:
            tx, ty = self.city.road.xy[node]
            t.waypoints = [(tx, ty)]

    def _lane_entry(self, route: list[int]) -> tuple[float, float] | None:
        """把消防车并回右侧车道：返回第一段车道线上离车最近的点。"""
        if len(route) < 2:
            return None
        (x1, y1), (x2, y2) = self.city.road.xy[route[0]], self.city.road.xy[route[1]]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 0.5:
            return None
        rx, ry = -dy / length, dx / length
        lx1, ly1 = x1 + rx * LANE, y1 + ry * LANE
        lx2, ly2 = x2 + rx * LANE, y2 + ry * LANE
        vx, vy = lx2 - lx1, ly2 - ly1
        den = vx * vx + vy * vy
        if den < 1e-6:
            return None
        k = ((self.truck.x - lx1) * vx + (self.truck.y - ly1) * vy) / den
        k = max(0.0, min(1.0, k))
        px, py = lx1 + vx * k, ly1 + vy * k
        if math.hypot(px - self.truck.x, py - self.truck.y) < 6.0:
            return None
        return (px, py)
    def dispatch_fire(self, fire: Fire) -> bool:
        t = self.truck
        planned = self._plan_route(fire.anchor_node)
        if planned is None:
            self._log(f"无法规划到火情 #{fire.id} 的路线，请检查路障", "critical")
            fire.checked = True
            fire.unreachable = True
            return False
        route, _length = planned
        t.state = "to_fire"
        t.spray = False
        self._set_route(route, fire.anchor_node)
        t.target_fire = fire.id
        self._log(f"自检通过，出发前往火情 #{fire.id}", "info")
        fire.checked = True
        return True

    def _plan_to_fire(self, fire: Fire) -> None:
        planned = self._plan_route(fire.anchor_node)
        if planned:
            self._set_route(planned[0], fire.anchor_node)
            self.truck.target_fire = fire.id

    def _nearest_station(self, kind: str) -> tuple[int, list[int], float] | None:
        best = None
        for idx, st in enumerate(self.city.stations):
            if st["kind"] != kind:
                continue
            planned = self._plan_route(st["node"])
            if planned is None:
                continue
            if best is None or planned[1] < best[2]:
                best = (idx, planned[0], planned[1])
        return best

    def dispatch_refill(self, resume_fire: int | None = None) -> bool:
        t = self.truck
        if self.ai_mission:
            self._end_ai_mission("水量偏低，转入补水流程")
        best = self._nearest_station("refill")
        if best is None:
            self._log("所有补水站都不可达，等待人工处理", "critical")
            return False
        idx, route, _length = best
        st = self.city.stations[idx]
        t.state = "to_refill"
        t.target_station = idx
        t.target_fire = resume_fire
        t.spray = False
        self._set_route(route, st["node"])
        t.target_station = idx
        if resume_fire:
            t.target_fire = resume_fire
        self._log(f"水量偏低，前往最近补水站：{st['name']}", "warning")
        return True

    def dispatch_repair(self) -> bool:
        t = self.truck
        if self.ai_mission:
            self._end_ai_mission("车辆故障，转入维修流程")
        best = self._nearest_station("repair")
        if best is None:
            self._log("维修站不可达，等待人工处理", "critical")
            return False
        idx, route, _length = best
        st = self.city.stations[idx]
        t.state = "to_repair"
        t.target_station = idx
        t.target_fire = None
        t.spray = False
        self._set_route(route, st["node"])
        t.target_station = idx
        self._log(f"检测到{FAULT_NAMES.get(t.fault or '', '功能')}故障，前往{st['name']}", "critical")
        return True

    # ------------------------------------------------------------------ AI 最优路线
    def _ai_targets(self) -> list[dict]:
        """当前值得跑一趟的目标点：有火情就去火情，没有就做全域巡检。"""
        targets = []
        for fire in sorted(self.fires, key=lambda f: f.id):
            if fire.intensity <= 0:
                continue
            targets.append({
                "node": fire.anchor_node,
                "kind": "fire",
                "label": f"火情 #{fire.id}",
                "fire_id": fire.id,
                "intensity": round(fire.intensity),
            })
        if not targets:
            for i, spot in enumerate(self.city.fire_spots):
                targets.append({
                    "node": spot["node"],
                    "kind": "patrol",
                    "label": f"巡检点 {i + 1}",
                })
        return targets

    def plan_ai_route(self) -> dict:
        """只做规划，不改变车辆状态；供按钮点击和「执行」前的预演使用。"""
        t = self.truck
        if self.self_check["running"]:
            return {"ok": False, "reason": "全车自检进行中，稍后再规划路线"}
        start = self._nearest_free_node(t.x, t.y, self._heavy_blocked_nodes())
        targets = self._ai_targets()
        adj = self._nav_adj()
        result = plan_multi_route(
            adj, self.city.road.xy, start, targets,
            start_point=(round(t.x, 1), round(t.y, 1)),
        )
        if not result.get("ok"):
            return result

        has_fire = any(x["kind"] == "fire" for x in targets)
        result["mode"] = "fire" if has_fire else "patrol"
        result["mode_label"] = "多火情最优处置路线" if has_fire else "全域预防性巡检路线"
        result["blocked_nodes"] = sorted(self._heavy_blocked_nodes())
        result["polyline"] = [[round(t.x, 1), round(t.y, 1)]] + [
            [round(x, 1), round(y, 1)]
            for x, y in self._lane_waypoints(result["path_nodes"])
        ]
        result["truck"] = {"x": round(t.x, 1), "y": round(t.y, 1), "state": t.state}
        result["explanation"] = [
            f"识别到 {result['algorithm']['targets']} 个目标点，先用 A* 算出两两最短路，"
            f"再用 {result['algorithm']['optimizer']} 决定访问顺序。",
            f"最优总里程 {result['total_distance']:.0f}，按发现顺序要跑 "
            f"{result['baseline_distance']:.0f}，节省 {result['saved_distance']:.0f}"
            f"（{result['saved_percent']:.1f}%）。",
            f"全程 {len(result['stops'])} 站，预计 {result['eta']:.0f} 秒跑完。",
        ]
        return result

    def execute_ai_route(self) -> dict:
        """接管车辆，按 AI 规划的顺序出车。"""
        t = self.truck
        if self.ai_mission:
            return {"ok": False, "msg": "AI 路线正在执行中"}
        if self.self_check["running"] or t.state == "self_check":
            return {"ok": False, "msg": "自检进行中，请稍后再接管"}
        plan = self.plan_ai_route()
        if not plan.get("ok"):
            return plan
        if not plan["stops"]:
            return {"ok": False, "msg": "没有可执行的目标点"}

        self.ai_mission = {
            "mode": plan["mode"],
            "plan": plan,
            "index": 0,
            "visited": [],            # 已完成站点的 order（兼容旧面板）
            "visited_nodes": [],      # 已完成站点的道路节点，重新规划时用于去重
            "completed": 0,
            "replans": 0,
            "last_replan": None,
            "replan_reason": "",
            "blocked_targets": [],
            "deferred": 0,
            "deferred_labels": [],
            "dwell": 0.0,
            "started_at": round(self.time, 1),
        }
        self.stats["ai_routes"] += 1
        t.state = "ai_mission"
        t.spray = False
        t.target_fire = None
        t.target_station = None
        self._log(f"AI 接管：{plan['mode_label']}，共 {len(plan['stops'])} 站，"
                  f"预计 {plan['eta']:.0f} 秒", "info")
        self._ai_goto(0)
        return {"ok": True, "mode": plan["mode"], "mode_label": plan["mode_label"],
                "stops": len(plan["stops"]), "total_distance": plan["total_distance"],
                "eta": plan["eta"], "algorithm": plan["algorithm"]["name"]}

    def _ai_goto(self, index: int) -> None:
        mission = self.ai_mission
        if not mission:
            return
        stops = mission["plan"]["stops"]
        if index >= len(stops):
            blocked = mission.get("blocked_targets") or []
            if blocked:
                self._end_ai_mission(f"其余 {len(blocked)} 个目标被封锁")
            else:
                self._end_ai_mission("AI 路线全部完成")
            return
        mission["index"] = index
        mission["dwell"] = 0.0
        stop = stops[index]
        planned = self._plan_route(stop["node"])
        if planned is None:
            self._ai_defer(index, "路线被不可移开的障碍物封锁")
            return

        route, _length = planned
        t = self.truck
        t.state = "ai_mission"
        t.spray = False
        t.goal_node = stop["node"]
        t.route = list(route)
        t.waypoints = self._lane_waypoints(route)
        t.wp_index = 0
        if not t.waypoints:
            tx, ty = self.city.road.xy[stop["node"]]
            t.waypoints = [(tx, ty)]
        t.target_fire = stop.get("fire_id") if stop["kind"] == "fire" else None
        t.target_station = None
        self._log(f"AI 导航：前往第 {index + 1}/{len(stops)} 站 {stop['label']}", "info")

    def _ai_defer(self, index: int, reason: str) -> None:
        """当前目标被完全封锁：先挂起，继续处理其他目标，结束时不谎报完成。"""
        mission = self.ai_mission
        if not mission:
            return
        stops = mission["plan"]["stops"]
        stop = stops[index]
        mission["deferred"] = mission.get("deferred", 0) + 1
        mission.setdefault("deferred_labels", []).append(stop["label"])
        self._log(f"AI 暂时无法到达 {stop['label']}（{reason}），先处理其他目标", "critical")
        self._ai_goto(index + 1)

    def _ai_replan(self, reason: str) -> bool:
        """FOV 反馈 / 新火情触发：AI 从当前位置重排剩余站点，任务不中断。"""
        mission = self.ai_mission
        if not mission:
            return False
        if self.time - self._ai_replan_at < AI_REPLAN_CD:
            return False
        self._ai_replan_at = self.time
        t = self.truck
        start = self._nearest_free_node(t.x, t.y, self._heavy_blocked_nodes())
        done = set(mission.get("visited_nodes", []))
        targets = [item for item in self._ai_targets() if item["node"] not in done]
        if not targets:
            self._end_ai_mission(f"{reason}：没有剩余目标")
            return True
        result = plan_multi_route(
            self._nav_adj(), self.city.road.xy, start, targets,
            start_point=(round(t.x, 1), round(t.y, 1)),
        )
        if not result.get("ok"):
            self._log(f"AI 重新规划失败：{result.get('reason', '无可行路线')}", "critical")
            self._end_ai_mission(f"{reason}：剩余目标都被封锁")
            return True

        has_fire = any(x["kind"] == "fire" for x in targets)
        result["mode"] = "fire" if has_fire else "patrol"
        result["mode_label"] = "多火情最优处置路线" if has_fire else "全域预防性巡检路线"
        result["blocked_nodes"] = sorted(self._heavy_blocked_nodes())
        result["polyline"] = [[round(t.x, 1), round(t.y, 1)]] + [
            [round(x, 1), round(y, 1)]
            for x, y in self._lane_waypoints(result["path_nodes"])
        ]
        mission["plan"] = result
        mission["index"] = 0
        mission["visited"] = []
        mission["blocked_targets"] = list(result.get("dropped") or [])
        mission["replans"] = mission.get("replans", 0) + 1
        mission["last_replan"] = round(self.time, 1)
        mission["replan_reason"] = reason
        self._fov_replans += 1
        self._fov_last_replan = round(self.time, 1)
        self._log(f"AI 按 {reason} 重新规划：剩余 {len(result['stops'])} 站，"
                  f"总里程 {result['total_distance']:.0f} 米", "warning")
        if mission["blocked_targets"]:
            self._log("这些目标被不可移开的障碍物封锁，暂时无法前往："
                      + "、".join(mission["blocked_targets"]), "critical")
        self._ai_goto(0)
        return True

    def _check_ai_arrival(self) -> None:
        mission = self.ai_mission
        if not mission:
            return
        stops = mission["plan"]["stops"]
        idx = mission["index"]
        if idx >= len(stops):
            self._end_ai_mission("AI 路线全部完成")
            return
        stop = stops[idx]
        ax, ay = self.city.road.xy[stop["node"]]
        t = self.truck
        if self._dist(t.x, t.y, ax, ay) > AI_ARRIVE:
            return
        if stop["kind"] == "fire":
            fire = self._fire_by_id(stop.get("fire_id"))
            if fire is None or fire.intensity <= 0:
                self._advance_ai(f"{stop['label']} 已无火情，继续前往下一站")
                return
            t.state = "extinguishing"
            t.speed = 0.0
            t.spray = True
            t.target_fire = fire.id
            self._log(f"到达 {stop['label']}，开始喷水灭火", "info")
        else:
            t.state = "ai_scan"
            t.speed = 0.0
            self._log(f"到达 {stop['label']}，正在扫描街区", "info")

    def _ai_scan(self, dt: float) -> None:
        mission = self.ai_mission
        if not mission:
            self._enter_idle()
            return
        mission["dwell"] += dt
        if mission["dwell"] >= AI_SCAN_TIME:
            idx = mission["index"]
            label = mission["plan"]["stops"][idx]["label"] if idx < len(mission["plan"]["stops"]) else "巡检点"
            self._advance_ai(f"{label} 巡检完成，未发现异常")

    def _advance_ai(self, message: str = "") -> None:
        mission = self.ai_mission
        if not mission:
            return
        if message:
            self._log(message, "info")
        idx = mission["index"]
        stops = mission["plan"]["stops"]
        if 0 <= idx < len(stops):
            mission["visited"].append(stops[idx]["order"])
            mission.setdefault("visited_nodes", []).append(stops[idx]["node"])
            mission["completed"] = mission.get("completed", 0) + 1
            self.stats["ai_stops"] += 1
        self._ai_goto(idx + 1)

    def _end_ai_mission(self, reason: str = "") -> None:
        mission = self.ai_mission
        if not mission:
            return
        self.ai_mission = None
        done = mission.get("completed", len(mission["visited"]))
        blocked = len(mission.get("blocked_targets") or [])
        deferred = mission.get("deferred", 0)
        level = "success" if not blocked and not deferred else "warning"
        extra = f"，{blocked + deferred} 个目标受阻待处理" if (blocked or deferred) else ""
        self._log(f"AI 路线结束：{reason}（完成 {done} 站{extra}）", level)
        if self.truck.state in ("ai_mission", "ai_scan", "extinguishing"):
            self._enter_idle()

    def cancel_ai(self) -> dict:
        if not self.ai_mission:
            return {"ok": False, "msg": "当前没有 AI 任务"}
        self._end_ai_mission("手动取消")
        return {"ok": True}

    def ai_status(self) -> dict:
        """给前端的心跳字段：当前 AI 任务进度与绘制用的整条路线。"""
        mission = self.ai_mission
        if not mission:
            return {
                "active": False, "mode": None, "mode_label": None,
                "index": 0, "total": 0, "visited": [],
                "stops": [], "polyline": [],
                "completed": 0, "replans": 0,
                "last_replan": None, "replan_reason": "",
                "blocked_targets": [], "deferred": 0,
                "fires": len(self.fires),
            }
        plan = mission["plan"]
        return {
            "active": True,
            "mode": plan["mode"],
            "mode_label": plan["mode_label"],
            "index": mission["index"],
            "total": len(plan["stops"]),
            "visited": list(mission["visited"]),
            "completed": mission.get("completed", len(mission["visited"])),
            "replans": mission.get("replans", 0),
            "last_replan": mission.get("last_replan"),
            "replan_reason": mission.get("replan_reason", ""),
            "blocked_targets": list(mission.get("blocked_targets") or []),
            "deferred": mission.get("deferred", 0),
            "blocked_nodes": plan.get("blocked_nodes", []),
            "total_distance": plan["total_distance"],
            "eta": plan["eta"],
            "stops": plan["stops"],
            "polyline": plan["polyline"],
            "fires": len(self.fires),
        }

    # ------------------------------------------------------------------ 自检
    def start_self_check(self, reason: str = "manual", fire_id: int | None = None) -> dict:
        if self.self_check["running"]:
            return {"ok": False, "msg": "自检正在进行中"}
        if self.ai_mission:
            self._end_ai_mission("开始全车自检")
        t = self.truck
        t.state = "self_check"
        t.speed = 0.0
        t.spray = False
        t.route = []
        t.waypoints = []
        t.wp_index = 0
        t.arm_target = None
        t.arm_progress = 0.0
        self._check_index = 0
        self._check_timer = 0.0
        self.self_check = self._empty_self_check()
        self.self_check.update({
            "running": True,
            "reason": reason,
            "fire_id": fire_id,
            "started_at": round(self.time, 1),
            "progress": 0.0,
        })
        self.stats["self_checks"] += 1
        label = {"fire": "火情触发", "fault": "故障触发", "manual": "手动触发"}.get(reason, reason)
        self._log(f"启动全车自检（{label}）", "info")
        return {"ok": True}

    def _update_self_check(self, dt: float) -> None:
        sc = self.self_check
        if not sc["running"]:
            return
        if self._check_index >= len(sc["steps"]):
            self._finish_self_check()
            return
        step = sc["steps"][self._check_index]
        if step["status"] == "pending":
            step["status"] = "running"
            self._check_timer = 0.0
        self._check_timer += dt
        if self._check_timer >= SELF_CHECK_STEP:
            self._evaluate_step(step)
            self._check_index += 1
            self._check_timer = 0.0
            sc["progress"] = round(self._check_index / len(sc["steps"]), 3)
            if self._check_index >= len(sc["steps"]):
                self._finish_self_check()

    def _evaluate_step(self, step: dict) -> None:
        t = self.truck
        key = step["key"]
        ok = True
        if key == "water":
            ok = t.water >= WATER_LOW
            step["detail"] = f"水量 {t.water:.0f}% / 阈值 {WATER_LOW:.0f}%"
            if not ok:
                step["detail"] += "，需要补水"
        elif key == "battery":
            ok = t.battery >= 20
            step["detail"] = f"电量 {t.battery:.0f}%"
            if not ok:
                step["detail"] += "，需要充电"
        else:
            ok = t.fault != key
            step["detail"] = "工作正常" if ok else f"检测到{FAULT_NAMES[key]}故障"
        step["status"] = "pass" if ok else "fail"

    def _finish_self_check(self) -> None:
        sc = self.self_check
        if not sc["running"]:
            return
        passed = all(step["status"] == "pass" for step in sc["steps"])
        if self.truck.fault:
            passed = False
        sc["running"] = False
        sc["passed"] = passed
        sc["progress"] = 1.0
        sc["finished_at"] = round(self.time, 1)
        sc["steps"] = [dict(step) for step in sc["steps"]]
        if passed:
            self._log("自检全部通过，车辆状态正常", "success")
            reason, fire_id = sc["reason"], sc["fire_id"]
            if reason == "fire" and fire_id is not None:
                fire = self._fire_by_id(fire_id)
                if fire:
                    if not self.dispatch_fire(fire):
                        self._enter_idle()
                else:
                    self._enter_idle()
            else:
                self._enter_idle()
        else:
            failed = [s["name"] for s in sc["steps"] if s["status"] != "pass"]
            if not failed and self.truck.fault:
                failed = [FAULT_NAMES.get(self.truck.fault, "未知故障")]
            self._log("自检未通过：" + "、".join(failed), "critical")
            if self.truck.fault:
                self.dispatch_repair()
            elif self.truck.water < WATER_LOW:
                self.dispatch_refill()
            else:
                self._enter_idle()

    # ------------------------------------------------------------------ 故障 / 火情 / 障碍
    def inject_fault(self, kind: str | None = None) -> dict:
        if self.truck.fault:
            return {"ok": False, "msg": f"已存在故障：{FAULT_NAMES.get(self.truck.fault)}"}
        kind = kind or self.rng.choice(list(FAULT_NAMES))
        if kind not in FAULT_NAMES:
            return {"ok": False, "msg": "未知故障类型"}
        self.truck.fault = kind
        self._log(f"车辆突发故障：{FAULT_NAMES[kind]}", "critical")
        if self.self_check["running"]:
            # 自检过程中突发故障：直接把对应项目标记为失败，页面结果保持一致
            for step in self.self_check["steps"]:
                if step["key"] == kind:
                    step["status"] = "fail"
                    step["detail"] = f"检测到{FAULT_NAMES[kind]}故障"
        if not self.self_check["running"]:
            if self.truck.state == "idle":
                self.start_self_check("fault")
            elif self.truck.state not in ("to_repair", "repairing", "self_check"):
                self.dispatch_repair()
        return {"ok": True, "fault": kind}

    def spawn_fire(self) -> dict:
        used = {(round(f.x, 1), round(f.y, 1)) for f in self.fires}
        free = [s for s in self.city.fire_spots if (round(s["x"], 1), round(s["y"], 1)) not in used]
        if not free:
            return {"ok": False, "msg": "所有点位都已有火情"}
        spot = self.rng.choice(free)
        fire = Fire(id=self._next_fire_id, x=spot["x"], y=spot["y"],
                    anchor_node=spot["node"])
        self._next_fire_id += 1
        self.fires.append(fire)
        if self.ai_mission:
            # AI 执行途中发现新火情：不中断任务，直接把新目标并入路线
            self._log(f"传感器检测到火情 #{fire.id}，AI 任务中，正在并入路线", "critical")
            self._ai_replan(f"新火情 #{fire.id}")
        else:
            self._log(f"传感器检测到火情 #{fire.id}，立即启动自检", "critical")
            if self.truck.state == "idle" and not self.self_check["running"]:
                self.start_self_check("fire", fire.id)
        return {"ok": True, "fire_id": fire.id}

    def spawn_obstacle(self, ahead: bool = False, kind: str = "barrier") -> dict:
        kind = "heavy" if kind == "heavy" else "barrier"
        if len(self.obstacles) >= 5:
            return {"ok": False, "msg": "路障太多了"}
        if kind == "heavy":
            if len([o for o in self.obstacles if o.immovable]) >= 3:
                return {"ok": False, "msg": "不可移开的路障太多了"}
        t = self.truck
        if ahead:
            # 放在机械臂够不到、但视野能覆盖的位置，正好演示「FOV 发现 → AI 绕行」
            dist = 170 if kind == "heavy" else 120
            x = t.x + math.cos(t.heading) * dist
            y = t.y + math.sin(t.heading) * dist
            x = max(30.0, min(self.city.WIDTH - 30.0, x))
            y = max(30.0, min(self.city.HEIGHT - 30.0, y))
        else:
            if self.rng.random() < 0.5:
                x = self.rng.choice(self.city.V_ROADS) + self.rng.choice((-LANE, LANE))
                y = self.rng.uniform(30, self.city.HEIGHT - 30)
            else:
                y = self.rng.choice(self.city.H_ROADS) + self.rng.choice((-LANE, LANE))
                x = self.rng.uniform(30, self.city.WIDTH - 30)
            if self._dist(x, y, t.x, t.y) < 120:
                return {"ok": False, "msg": "离小车太近，换个位置"}
        obstacle = Obstacle(id=self._next_obstacle_id, x=x, y=y, kind=kind)
        self._next_obstacle_id += 1
        self.obstacles.append(obstacle)
        label = "不可移开路障" if kind == "heavy" else "路障"
        self._log(f"道路上出现{label} #{obstacle.id}", "warning")
        self._update_fov(0.0)
        return {"ok": True, "obstacle_id": obstacle.id, "kind": kind}

    # ------------------------------------------------------------------ 地图编辑器
    def spawn_jam_demo(self) -> dict:
        """演示用：在车头方向最近的路口停两辆车，制造能被 FOV 发现的拥堵。"""
        t = self.truck
        best, best_score = None, float("inf")
        intersections = [{"x": x, "y": y}
                         for x in self.city.V_ROADS for y in self.city.H_ROADS]
        for p in intersections:
            dx, dy = p["x"] - t.x, p["y"] - t.y
            dist = math.hypot(dx, dy)
            if not (60.0 <= dist <= 300.0):
                continue
            diff = abs((math.atan2(dy, dx) - t.heading + math.pi) % (2 * math.pi) - math.pi)
            score = dist + diff * 140.0          # 优先选车头方向、视野内的路口
            if score < best_score:
                best, best_score = p, score
        if best is None:
            return {"ok": False, "msg": "附近没有适合制造拥堵的路口"}
        if len([c for c in self.cars if c.temporary]) >= 4:
            return {"ok": False, "msg": "演示拥堵还在，等它自己疏解"}
        for offset in (-18.0, 18.0):
            car = NpcCar(
                id=self._next_car_id,
                x=round(best["x"] + offset, 1), y=round(best["y"] + 8.0, 1),
                heading=math.pi / 2, cruise=0.0,
                color=self.rng.choice(CAR_COLORS),
                hold=JAM_DEMO_TIME, temporary=True,
            )
            self._next_car_id += 1
            self.cars.append(car)
        self._update_fov(0.0)
        self._log(f"演示：路口 ({best['x']:.0f}, {best['y']:.0f}) 两辆车停住，"
                  f"形成临时拥堵", "warning")
        return {"ok": True, "x": best["x"], "y": best["y"], "cars": 2}

    def _clamp_xy(self, x: float, y: float, margin: float = 20.0) -> tuple[float, float]:
        return (max(margin, min(self.city.WIDTH - margin, float(x))),
                max(margin, min(self.city.HEIGHT - margin, float(y))))

    def add_obstacle_at(self, x: float, y: float, kind: str = "barrier") -> dict:
        """地图编辑器：在任意位置放一个路障；kind=heavy 时是移不开的水泥墩。"""
        x, y = self._clamp_xy(x, y)
        kind = "heavy" if kind == "heavy" else "barrier"
        if len(self.obstacles) >= 12:
            return {"ok": False, "msg": "路障太多，请先清理一些"}
        if kind == "heavy" and len([o for o in self.obstacles if o.immovable]) >= 6:
            return {"ok": False, "msg": "不可移开的路障太多了，请先删除一些"}
        obstacle = Obstacle(id=self._next_obstacle_id, x=round(x, 1),
                            y=round(y, 1), custom=True, kind=kind)
        self._next_obstacle_id += 1
        self.obstacles.append(obstacle)
        self._update_fov(0.0)
        label = "不可移开路障" if kind == "heavy" else "路障"
        self._log(f"地图编辑器放置{label} #{obstacle.id}", "warning")
        if kind == "heavy" and self.ai_mission:
            self._ai_replan(f"地图新增不可移开路障 #{obstacle.id}")
        return {"ok": True, "obstacle_id": obstacle.id, "kind": kind,
                "x": obstacle.x, "y": obstacle.y}

    def add_fire_at(self, x: float, y: float) -> dict:
        """地图编辑器：新增火情。

        火点会绑定最近的道路节点供消防车停靠；如果点击位置离道路太远，
        就把火点拉回到 50 单位以内，保证到达后一定在 100 的喷水范围内。
        """
        x, y = self._clamp_xy(x, y)
        if any(math.hypot(f.x - x, f.y - y) < 30 for f in self.fires):
            return {"ok": False, "msg": "这里已经有火情了"}
        anchor = self.city.nearest_road_node(x, y)
        ax, ay = self.city.road.xy[anchor]
        dx, dy = x - ax, y - ay
        dist = math.hypot(dx, dy)
        if dist > 50.0:
            x, y = ax + dx / dist * 50.0, ay + dy / dist * 50.0
        fire = Fire(id=self._next_fire_id, x=round(x, 1), y=round(y, 1),
                    anchor_node=anchor, checked=False, custom=True)
        self._next_fire_id += 1
        self.fires.append(fire)
        self._log(f"地图编辑器新增火情 #{fire.id}", "critical")
        if self.ai_mission:
            self._ai_replan(f"新火情 #{fire.id}")
        elif self.truck.state == "idle" and not self.self_check["running"]:
            self.start_self_check("fire", fire.id)
        return {"ok": True, "fire_id": fire.id, "x": fire.x, "y": fire.y}

    def add_pedestrian_at(self, x: float, y: float) -> dict:
        """地图编辑器：在最近的人行道节点生成一名行人。"""
        x, y = self._clamp_xy(x, y)
        if len(self.pedestrians) >= 40:
            return {"ok": False, "msg": "行人太多了，请先清理一些"}
        start = self.city.nearest_ped_node(x, y)
        for _attempt in range(12):
            goal = self.city.random_ped_node(self.rng)
            if goal == start:
                continue
            path = self.city.ped_route(start, goal)
            if len(path) < 2:
                continue
            px, py = self.city.ped.xy[start]
            ped = Pedestrian(
                id=self._next_ped_id, x=px, y=py,
                speed=self.rng.uniform(20, 30),
                emoji=self.rng.choice(["🚶", "🚶", "🚶‍♂️", "🚶‍♀️"]),
                path=path, idx=1, custom=True,
            )
            self._next_ped_id += 1
            self._maybe_wait(ped)
            self.pedestrians.append(ped)
            self._log(f"地图编辑器新增行人 #{ped.id}", "info")
            return {"ok": True, "pedestrian_id": ped.id,
                    "x": round(px, 1), "y": round(py, 1)}
        return {"ok": False, "msg": "附近没有可用的人行道路线"}

    def erase_at(self, x: float, y: float, radius: float = 38.0) -> dict:
        """地图编辑器：删除离点击位置最近的路障 / 火情 / 行人。"""
        x, y = self._clamp_xy(x, y)
        candidates: list[tuple[float, str, object]] = []
        for obstacle in self.obstacles:
            candidates.append((math.hypot(obstacle.x - x, obstacle.y - y),
                               "obstacle", obstacle))
        for fire in self.fires:
            candidates.append((math.hypot(fire.x - x, fire.y - y), "fire", fire))
        for ped in self.pedestrians:
            candidates.append((math.hypot(ped.x - x, ped.y - y), "pedestrian", ped))
        if not candidates:
            return {"ok": False, "msg": "这里没有可删除的动态对象"}
        dist, kind, target = min(candidates, key=lambda item: item[0])
        if dist > radius:
            return {"ok": False, "msg": "离对象太远，请点得更准一些"}
        if kind == "obstacle":
            self.obstacles = [o for o in self.obstacles if o.id != target.id]
            self._fov_known_obstacles.pop(target.id, None)
        elif kind == "fire":
            self.fires = [f for f in self.fires if f.id != target.id]
        else:
            self.pedestrians = [p for p in self.pedestrians if p.id != target.id]
        self._update_fov(0.0)
        self._log(f"地图编辑器删除{ {'obstacle': '路障', 'fire': '火情', 'pedestrian': '行人'}[kind] } "
                  f"#{target.id}", "info")
        return {"ok": True, "kind": kind, "id": target.id}

    def clear_dynamic(self) -> dict:
        """清空编辑器添加的内容：全部火情 / 路障 + 自定义行人。"""
        fires = len(self.fires)
        obstacles = len(self.obstacles)
        peds = len([p for p in self.pedestrians if p.custom])
        self.fires = []
        self.obstacles = []
        self.pedestrians = [p for p in self.pedestrians if not p.custom]
        self._fov_known_obstacles.clear()
        self._fov_hazards.clear()
        self._update_fov(0.0)
        self._log(f"地图编辑器清空：火情 {fires}、路障 {obstacles}、自定义行人 {peds}", "info")
        return {"ok": True, "fires": fires, "obstacles": obstacles, "pedestrians": peds}

    # ------------------------------------------------------------------ 行人
    def _spawn_pedestrian(self) -> None:
        for _attempt in range(8):
            start = self.city.random_ped_node(self.rng)
            goal = self.city.random_ped_node(self.rng)
            if start == goal:
                continue
            path = self.city.ped_route(start, goal)
            if len(path) < 2:
                continue
            x, y = self.city.ped.xy[start]
            emoji = self.rng.choice(["🚶", "🚶", "🚶‍♂️", "🚶‍♀️"])
            ped = Pedestrian(id=self._next_ped_id, x=x, y=y,
                             speed=self.rng.uniform(20, 30), emoji=emoji,
                             path=path, idx=1)
            self._next_ped_id += 1
            self._maybe_wait(ped)
            self.pedestrians.append(ped)
            return

    def _maybe_wait(self, ped: Pedestrian) -> None:
        if ped.idx >= len(ped.path):
            return
        kind = self.city.ped_edge_kind.get((ped.path[ped.idx - 1], ped.path[ped.idx]))
        if kind in ("cross_ew", "cross_ns") and not self._cross_allowed(kind):
            ped.waiting = True
            ped.wait_kind = kind
            ped.state = "waiting"
        else:
            ped.waiting = False
            ped.state = "crossing" if kind and kind.startswith("cross") else "walking"

    def _cross_allowed(self, kind: str) -> bool:
        if kind == "cross_ew":       # 东西向过街，和东西向车流同时放行
            return self.signal == "EW"
        if kind == "cross_ns":
            return self.signal == "NS"
        return True

    def _new_ped_route(self, ped: Pedestrian) -> None:
        goal = self.city.random_ped_node(self.rng)
        # 从「正在走向的那个节点」接着走；没有才取最近节点。
        # 绝不凭空跳到一个随机节点，否则就等于横穿马路。
        if ped.idx < len(ped.path):
            start = ped.path[ped.idx]
        else:
            start = self.city.nearest_ped_node(ped.x, ped.y)
        path = self.city.ped_route(start, goal)
        if len(path) < 2:
            return
        ped.path = path
        ped.idx = 1
        self._maybe_wait(ped)

    def _update_pedestrians(self, dt: float) -> None:
        for ped in self.pedestrians:
            if ped.waiting:
                if self._cross_allowed(ped.wait_kind):
                    ped.waiting = False
                    ped.state = "crossing"
                else:
                    continue
            if ped.idx >= len(ped.path):
                self._new_ped_route(ped)
                if ped.idx >= len(ped.path):
                    continue
            tx, ty = self.city.ped.xy[ped.path[ped.idx]]
            dx, dy = tx - ped.x, ty - ped.y
            dist = math.hypot(dx, dy)
            if dist < 3:
                ped.x, ped.y = tx, ty
                ped.idx += 1
                if ped.idx >= len(ped.path):
                    self._new_ped_route(ped)
                else:
                    self._maybe_wait(ped)
                continue
            step = min(dist, ped.speed * dt)
            nx = ped.x + dx / dist * step
            ny = ped.y + dy / dist * step
            if self._ped_blocked(ped, nx, ny):
                ped.state = "waiting_car"          # 车还没过去，先站住
                continue
            ped.x, ped.y = nx, ny

    def _ped_blocked(self, ped: Pedestrian, nx: float, ny: float) -> bool:
        """行人这一步会不会撞到车：会就站着等，等车过去再走。"""
        if math.hypot(self.truck.x - nx, self.truck.y - ny) < 34:
            if math.hypot(self.truck.x - nx, self.truck.y - ny) < math.hypot(
                    self.truck.x - ped.x, self.truck.y - ped.y):
                return True
        for car in self.cars:
            cx, cy = self._car_drawn_xy(car)
            d = math.hypot(cx - nx, cy - ny)
            if d < 26 and d < math.hypot(cx - ped.x, cy - ped.y):
                return True
        return False

    # ------------------------------------------------------------------ 私家车（NPC）
    def spawn_car(self) -> dict:
        """在路网随机位置放一辆私家车，路上常驻 CAR_MAX 辆。"""
        if len(self.cars) >= CAR_MAX:
            return {"ok": False, "msg": f"路上最多 {CAR_MAX} 辆小车"}
        nodes = self.city.road.xy
        fallback = None
        for _attempt in range(24):
            start = self.rng.randrange(len(nodes))
            goal = self.rng.randrange(len(nodes))
            if start == goal:
                continue
            sx, sy = nodes[start]
            if self._dist(sx, sy, self.truck.x, self.truck.y) < CAR_SPAWN_CLEAR:
                continue
            if any(self._dist(sx, sy, c.x, c.y) < 70 for c in self.cars):
                continue
            route = graph_astar(self.city.road.adj, start, goal, nodes)
            if len(route) < 3:
                continue
            waypoints = self._lane_waypoints(route)
            if not waypoints:
                continue
            length = sum(self._dist(*nodes[a], *nodes[b])
                         for a, b in zip(route, route[1:]))
            if length >= CAR_ROUTE_MIN:
                return self._add_car(start, route, waypoints)
            if fallback is None:
                fallback = (start, route, waypoints)
        if fallback is not None:
            return self._add_car(*fallback)
        return {"ok": False, "msg": "暂时没有合适的出生点"}

    def _add_car(self, start: int, route: list[int],
                 waypoints: list[tuple[float, float]]) -> dict:
        sx, sy = self.city.road.xy[start]
        car = NpcCar(
            id=self._next_car_id, x=sx, y=sy,
            heading=math.atan2(waypoints[0][1] - sy, waypoints[0][0] - sx),
            cruise=self.rng.uniform(CAR_SPEED_MIN, CAR_SPEED_MAX),
            color=self.rng.choice(CAR_COLORS),
            route=route, waypoints=waypoints, target_node=route[-1],
        )
        self._next_car_id += 1
        self.cars.append(car)
        self._log(f"小车 #{car.id} 上路（路上共 {len(self.cars)}/{CAR_MAX} 辆）", "info")
        return {"ok": True, "car_id": car.id}

    def _truck_busy(self) -> bool:
        """消防车是否正在出车途中；只有出车时私家车才需要让行。"""
        return self.truck.state in ("to_fire", "to_refill", "to_repair", "ai_mission")

    def _update_cars(self, dt: float) -> None:
        expired: list[int] = []
        for car in self.cars:
            if car.temporary:
                # 拥堵演示车：停住一段时间后自动消失，不参与交通
                car.hold -= dt
                car.speed = 0.0
                if car.hold <= 0:
                    expired.append(car.id)
                continue
            if car.escape > 0:
                car.escape = max(0.0, car.escape - dt)
            if car.pull_aside > 0:
                car.pull_aside = max(0.0, car.pull_aside - dt)
            self._car_yield(car)
            self._car_drive(car, dt)
            if car.wp_index >= len(car.waypoints):
                # 常驻：到终点不停驶离，换一条随机路线继续跑
                car.speed = 0.0
                car.retry -= dt
                if car.retry <= 0:
                    car.retry = 1.0
                    self._car_new_route(car)
        if expired:
            self.cars = [c for c in self.cars if c.id not in expired]
            self._log("路口临时拥堵已自行疏解", "info")

    def _car_new_route(self, car: NpcCar, blocked: tuple[int, ...] = (),
                       uturn: bool = False) -> bool:
        """给常驻小车换一条新路线：默认从车头前方的节点出发，尽量不掉头。

        blocked 里的节点（比如堵住的路口）不许经过；uturn=True 时允许掉头
        （用于路口堵死、需要原路退出去的情况）。
        """
        nodes = self.city.road.xy
        ahead = (car.x + math.cos(car.heading) * 45.0,
                 car.y + math.sin(car.heading) * 45.0)
        start = self.city.nearest_road_node(*ahead)
        if start in blocked:
            start = self.city.nearest_road_node(car.x, car.y)
        if start in blocked:                       # 就堵在路口里：找最近的非禁行节点
            others = [nid for nid in range(len(nodes)) if nid not in blocked]
            start = min(others, key=lambda nid: self._dist(car.x, car.y, *nodes[nid]))
        taken = {o.target_node for o in self.cars if o.id != car.id}
        fallback = None
        for _attempt in range(12):
            goal = self.rng.randrange(len(nodes))
            if goal == start or goal in taken or goal in blocked:
                continue
            route = graph_astar(self.city.road.adj, start, goal, nodes, blocked=blocked)
            if not route or len(route) < 3:
                continue
            waypoints = self._lane_waypoints(route)
            if not waypoints:
                continue
            if fallback is None:
                fallback = (route, waypoints, goal)
            # 第一段朝车尾方向（掉头）就换终点；正常左转右转都允许。
            # 需要掉头撤离时（uturn=True）就不做这个限制。
            if not uturn:
                dx, dy = waypoints[0][0] - car.x, waypoints[0][1] - car.y
                if math.hypot(dx, dy) > 1e-3:
                    diff = abs((math.atan2(dy, dx) - car.heading + math.pi) % (2 * math.pi) - math.pi)
                    if diff > 1.75:
                        continue
            self._apply_car_route(car, route, waypoints, goal)
            return True
        if fallback is not None:                   # 实在挑不到合适的，就用第一条
            self._apply_car_route(car, *fallback)
            return True
        return False

    def _apply_car_route(self, car: NpcCar, route: list[int],
                         waypoints: list[tuple[float, float]], goal: int) -> None:
        car.route = route
        car.waypoints = waypoints
        car.wp_index = 0
        car.target_node = goal

    def _truck_oncoming(self, car: NpcCar) -> bool:
        """消防车是不是迎面驶来（车头方向大致相反）。"""
        diff = (self.truck.heading - car.heading + math.pi) % (2 * math.pi) - math.pi
        return abs(diff) > 1.75

    def _car_drawn_xy(self, car: NpcCar) -> tuple[float, float]:
        """小车的实际绘制位置（车道位置 + 横向让行偏移）。"""
        return (car.x - math.sin(car.heading) * car.side,
                car.y + math.cos(car.heading) * car.side)

    def _car_lane_heading(self, car: NpcCar) -> float | None:
        """小车所在路段的车道方向（用来把车头摆正）。"""
        i = car.wp_index
        if i < len(car.waypoints):
            tx, ty = car.waypoints[i]
            if i > 0:
                px, py = car.waypoints[i - 1]
            else:
                px, py = car.x, car.y
            if math.hypot(tx - px, ty - py) > 1.0:
                return math.atan2(ty - py, tx - px)
        if i > 0:
            px, py = car.waypoints[i - 1]
            tx, ty = car.waypoints[min(i, len(car.waypoints) - 1)]
            if math.hypot(tx - px, ty - py) > 1.0:
                return math.atan2(ty - py, tx - px)
        return None

    def _in_building(self, x: float, y: float, margin: float = 8.0) -> bool:
        """这个位置是不是压在建筑上：小车不许开进建筑。"""
        for b in self.city.buildings:
            if (b["x"] - margin <= x <= b["x"] + b["w"] + margin
                    and b["y"] - margin <= y <= b["y"] + b["h"] + margin):
                return True
        return False

    def _truck_gap(self, car: NpcCar) -> tuple[float, float] | None:
        """消防车相对小车的（纵向距离, 横向偏移）；只算车头前方的车。"""
        cx, cy = self._car_drawn_xy(car)
        dx, dy = self.truck.x - cx, self.truck.y - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return None
        rel = (math.atan2(dy, dx) - car.heading + math.pi) % (2 * math.pi) - math.pi
        longi = dist * math.cos(rel)
        if longi <= 0:
            return None
        return longi, abs(dist * math.sin(rel))

    def _car_yield(self, car: NpcCar) -> None:
        """行驶准则：遇到消防车就靠边停下，把车道让出来；路口另外排队。

        - 消防车在附近（不论迎面、同向、还是停在旁边）：靠路肩停下；
        - 到了路口：一次只放一辆车进去，按编号先来后到；
        - 消防车被堵 / 故障时：这一段先不拦，让小车自己开走。
        """
        if car.escape > 0:
            # 消防车被堵 / 故障：这一段先让小车自己开走，不再等它
            car.state = "driving"
            return

        cx, cy = self._car_drawn_xy(car)
        dist = self._dist(cx, cy, self.truck.x, self.truck.y)
        limit = CAR_CLEAR_DIST if car.yielding else CAR_YIELD_DIST
        if dist <= limit or car.pull_aside > 0:
            car.state = "yield_roadside"           # 靠边停下，让消防车先过
            return

        if self._junction_busy(car):
            car.state = "yield_intersection"
            return

        car.state = "driving"

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """两个朝向的夹角（-π..π）。"""
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def _junction_busy(self, car: NpcCar) -> bool:
        """路口一次只放一辆车：路口里有车、或有编号更小的车先到，就在路口外等。"""
        node = self._intersection_ahead(car)
        if node is None:
            return False
        nx, ny = self.city.road.xy[node]
        mine = self._dist(car.x, car.y, nx, ny)
        if mine < 45.0:
            return False                           # 已经在路口里：先把路口走完
        for other in self.cars:
            if other.id == car.id:
                continue
            ox, oy = self._car_drawn_xy(other)
            d = self._dist(ox, oy, nx, ny)
            if d < 55.0:
                return True                        # 路口里已经有车
            if d >= CAR_INTERSECTION_DIST:
                continue
            if abs(self._angle_diff(other.heading, car.heading)) < 0.7:
                continue                           # 同向的不算抢行
            if other.id < car.id:
                return True                        # 编号小的先过，不会互相死等
        return False

    def _truck_in_intersection(self, car: NpcCar) -> bool:
        """小车快到路口，而且消防车正接近 / 通过同一个路口。"""
        node = self._intersection_ahead(car)
        if node is None:
            return False
        nx, ny = self.city.road.xy[node]
        return self._dist(self.truck.x, self.truck.y, nx, ny) < 100.0

    def _intersection_ahead(self, car: NpcCar) -> int | None:
        """车辆正前方最近的路口；只有进入 CAR_INTERSECTION_DIST 才算「在路口」。"""
        for node in car.route[car.wp_index + 1:]:
            if not self.city.is_intersection(node):
                continue
            nx, ny = self.city.road.xy[node]
            if self._dist(car.x, car.y, nx, ny) <= CAR_INTERSECTION_DIST:
                return node
            return None                          # 最近的路口还远，后面的更不用看
        return None

    def _car_target_side(self, car: NpcCar) -> float:
        """让行时的横向目标：+ 靠路肩；车头还没顺正时先不横移，免得扫到对向车道。"""
        if car.state != "yield_roadside":
            return 0.0
        if car.wp_index < len(car.waypoints):
            tx, ty = car.waypoints[car.wp_index]
            if math.hypot(tx - car.x, ty - car.y) > 1e-3:
                if abs((math.atan2(ty - car.y, tx - car.x) - car.heading + math.pi)
                       % (2 * math.pi) - math.pi) > 0.8:
                    return 0.0
        return CAR_PULL_OVER

    def _car_target_speed(self, car: NpcCar) -> float:
        if car.escape > 0:
            return car.cruise                      # 正在撤离：全速离开
        if car.state == "yield_intersection":
            return 0.0
        if car.state == "yield_roadside":
            # 先减速向路肩靠，贴边就停住，等消防车过去
            if car.side >= CAR_PULL_OVER - 0.5:
                return 0.0
            if not self._side_ok(car, min(CAR_PULL_OVER, car.side + 1.0)):
                return 0.0                          # 再靠就压到建筑了，就停在这儿
            return min(car.cruise, 30.0)
        if car.state == "wait_truck":
            # 同侧前方消防车停下了：开到它后面停住，等它起步
            gap = self._truck_gap(car)
            if gap is None or gap[0] < CAR_WAIT_GAP:
                return 0.0
            return min(car.cruise, 28.0)
        gap = self._truck_gap(car)
        if gap is not None and gap[1] < 36 and gap[0] < 120:
            # 同侧前方是行驶中的消防车：按它的车速跟车，保持车距
            return min(car.cruise, max(self.truck.speed, 0.0))
        return car.cruise

    def _oncoming_speed(self, car: NpcCar) -> float:
        """会车：对向车辆窜到自己车道上时减速 / 停车，避免迎面相撞。"""
        cap = car.cruise
        cx, cy = self._car_drawn_xy(car)
        for other in self.cars:
            if other.id == car.id:
                continue
            if abs(self._angle_diff(other.heading, car.heading)) < 1.6:
                continue                            # 同向不算会车
            ox, oy = self._car_drawn_xy(other)
            dx, dy = ox - cx, oy - cy
            dist = math.hypot(dx, dy)
            if dist > 90 or dist < 1e-3:
                continue
            rel = (math.atan2(dy, dx) - car.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(rel) > 0.6:
                continue
            gap = dist * math.cos(rel)
            if gap <= 0 or abs(dist * math.sin(rel)) > 30:
                continue                            # 各在各的车道，互不影响
            if gap < 30:
                cap = 0.0
            elif gap < 55:
                cap = min(cap, 24.0)
        return cap

    def _pedestrian_speed(self, car: NpcCar) -> float:
        """前方有行人（过街 / 走上车道）：减速，近了就停，等人过去再走。"""
        cap = car.cruise
        cx, cy = self._car_drawn_xy(car)
        for ped in self.pedestrians:
            dx, dy = ped.x - cx, ped.y - cy
            dist = math.hypot(dx, dy)
            if dist > 80 or dist < 1e-3:
                continue
            rel = (math.atan2(dy, dx) - car.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(rel) > 1.0:
                continue                            # 正前方 ±57°
            if abs(dist * math.sin(rel)) > 24:
                continue                            # 不在本车道上（比如在人行道走）
            gap = dist * math.cos(rel)
            if gap <= 0:
                continue
            if gap < 56:
                cap = 0.0
            elif gap < 80:
                cap = min(cap, 20.0)
        return cap

    def _car_drive(self, car: NpcCar, dt: float) -> None:
        wants = self._car_target_speed(car) > 1.0
        target_speed = min(self._car_target_speed(car), self._car_follow_speed(car),
                           self._oncoming_speed(car), self._pedestrian_speed(car))
        old_heading = car.heading
        if car.wp_index < len(car.waypoints):
            tx, ty = car.waypoints[car.wp_index]
            dx, dy = tx - car.x, ty - car.y
            dist = math.hypot(dx, dy)
            if dist < 12:
                car.wp_index += 1
            elif dist > 0:
                desired = math.atan2(dy, dx)
                diff = (desired - car.heading + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) > 0.95:
                    target_speed = min(target_speed, 26.0)
                if car.speed > 3:
                    car.heading += max(-CAR_TURN * dt, min(CAR_TURN * dt, diff))
        else:
            target_speed = 0.0

        if car.speed < target_speed:
            car.speed = min(target_speed, car.speed + CAR_ACCEL * dt)
        else:
            car.speed = max(target_speed, car.speed - CAR_BRAKE * dt)

        # 想走却一动不动：累计「堵住」时间，够久就倒车让一步（有空间才倒）
        if wants and car.speed < 1.0:
            car.stuck += dt
            if car.stuck > 2.0 and car.reverse <= 0:
                car.reverse = 1.2
                car.stuck = 0.0
        else:
            car.stuck = 0.0

        nx = car.x + math.cos(car.heading) * car.speed * dt
        ny = car.y + math.sin(car.heading) * car.speed * dt
        nx = max(10.0, min(self.city.WIDTH - 10.0, nx))
        ny = max(10.0, min(self.city.HEIGHT - 10.0, ny))
        car.heading = (car.heading + math.pi) % (2 * math.pi) - math.pi
        px = nx - math.sin(car.heading) * car.side
        py = ny + math.cos(car.heading) * car.side
        if self._would_crowd(car, nx, ny) or self._in_building(px, py):
            car.speed = 0.0                        # 前方没空间 / 要压到建筑：原地等
            car.heading = old_heading              # 也别再打方向，免得原地打转
        else:
            car.x, car.y = nx, ny

        if car.reverse > 0:
            car.reverse -= dt
            back_speed = 24.0
            bx = car.x - math.cos(car.heading) * back_speed * dt
            by = car.y - math.sin(car.heading) * back_speed * dt
            bx = max(10.0, min(self.city.WIDTH - 10.0, bx))
            by = max(10.0, min(self.city.HEIGHT - 10.0, by))
            rpx = bx - math.sin(car.heading) * car.side
            rpy = by + math.cos(car.heading) * car.side
            if (not self._would_crowd(car, bx, by) and not self._truck_crowds(bx, by)
                    and not self._in_building(rpx, rpy)):
                car.x, car.y = bx, by
                car.reverse = 0.0                  # 倒一步就停，避免倒退撞到后车

        # 停住时把车头摆正：不许横在路面上（靠边让消防车时允许斜停在路边）
        if abs(car.speed) < 3.0 and car.side < CAR_PULL_OVER - 0.5:
            lane = self._car_lane_heading(car)
            if lane is not None:
                diff = (lane - car.heading + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) > 0.08:
                    turn = min(abs(diff), 4.0 * dt)
                    new_heading = car.heading + math.copysign(turn, diff)
                    px = car.x - math.sin(new_heading) * car.side
                    py = car.y + math.cos(new_heading) * car.side
                    if not self._in_building(px, py) and not self._rotated_crowds(car, px, py):
                        car.heading = (new_heading + math.pi) % (2 * math.pi) - math.pi

        # 横向让行：靠路肩，让完自动回正；横移同样不许挤到别人
        want = self._car_target_side(car)
        step = CAR_SIDE_RATE * dt
        if want > car.side:
            candidate = min(want, car.side + step)
        elif want < car.side:
            candidate = max(want, car.side - step)
        else:
            candidate = car.side
        if candidate != car.side and self._side_ok(car, candidate):
            car.side = candidate

    def _side_ok(self, car: NpcCar, new_side: float) -> bool:
        """横移这一步会不会挤到别人 / 压到建筑。"""
        px = car.x - math.sin(car.heading) * new_side
        py = car.y + math.cos(car.heading) * new_side
        cx, cy = self._car_drawn_xy(car)
        if self._in_building(px, py):
            return False                           # 小车不许开进建筑
        if self._truck_crowds(px, py) and self._closer(px, py, cx, cy, self.truck.x, self.truck.y):
            return False
        for other in self.cars:
            if other.id == car.id:
                continue
            ox, oy = self._car_drawn_xy(other)
            if self._cars_crowd(car, px, py, ox, oy) and self._closer(px, py, cx, cy, ox, oy):
                return False
        for ped in self.pedestrians:
            if math.hypot(ped.x - px, ped.y - py) < 24 and \
                    math.hypot(ped.x - px, ped.y - py) < math.hypot(ped.x - cx, ped.y - cy):
                return False                       # 别把车横移到行人身上
        return True

    def _rotated_crowds(self, car: NpcCar, px: float, py: float) -> bool:
        """原地摆正车头时，新车身位置会不会碰到别人。"""
        if self._truck_crowds(px, py):
            return True
        for other in self.cars:
            if other.id == car.id:
                continue
            ox, oy = self._car_drawn_xy(other)
            if self._cars_crowd(car, px, py, ox, oy):
                return True
        for ped in self.pedestrians:
            if math.hypot(ped.x - px, ped.y - py) < 22:
                return True
        return False
    def _would_crowd(self, car: NpcCar, nx: float, ny: float) -> bool:
        """这一步挪过去会不会挤到别人：会就这一帧原地不动（不推挤、不瞬移）。

        纵向留车距、横向留车宽，分开判断；已经在远离对方时放行，避免自己卡死。
        """
        px = nx - math.sin(car.heading) * car.side
        py = ny + math.cos(car.heading) * car.side
        cx, cy = self._car_drawn_xy(car)
        if self._truck_crowds(px, py) and self._closer(px, py, cx, cy, self.truck.x, self.truck.y):
            return True
        for other in self.cars:
            if other.id == car.id:
                continue
            ox, oy = self._car_drawn_xy(other)
            if not self._cars_crowd(car, px, py, ox, oy):
                continue
            if self._closer(px, py, cx, cy, ox, oy):
                return True
        return False

    def _truck_crowds(self, x: float, y: float) -> bool:
        """(x, y) 是否落进消防车的前后 / 左右禁区。"""
        dx, dy = x - self.truck.x, y - self.truck.y
        c, s = math.cos(self.truck.heading), math.sin(self.truck.heading)
        return abs(dx * c + dy * s) < CAR_TRUCK_LEN and abs(-dx * s + dy * c) < CAR_TRUCK_SIDE

    @staticmethod
    def _cars_crowd(car: NpcCar, x: float, y: float, ox: float, oy: float) -> bool:
        """(x, y) 是否贴进另一辆小车的车身范围。"""
        dx, dy = x - ox, y - oy
        c, s = math.cos(car.heading), math.sin(car.heading)
        return abs(dx * c + dy * s) < CAR_CAR_LEN and abs(-dx * s + dy * c) < CAR_CAR_SIDE

    @staticmethod
    def _closer(nx: float, ny: float, cx: float, cy: float, vx: float, vy: float) -> bool:
        """新位置是否比现在更靠近对方。"""
        return math.hypot(vx - nx, vy - ny) < math.hypot(vx - cx, vy - cy)

    def _car_follow_speed(self, car: NpcCar) -> float:
        """跟车：同车道前方有车就按车距平滑减速，始终保持安全距离。"""
        cap = car.cruise
        cx, cy = self._car_drawn_xy(car)
        for other in self.cars:
            if other.id == car.id:
                continue
            ox, oy = self._car_drawn_xy(other)
            dx, dy = ox - cx, oy - cy
            dist = math.hypot(dx, dy)
            if dist > 110 or dist < 0.1:
                continue
            rel = (math.atan2(dy, dx) - car.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(dist * math.sin(rel)) > 26:
                continue                       # 不在同一车道，互不影响
            gap = dist * math.cos(rel)
            if gap <= 0:
                continue                       # 在自己后面
            cap = min(cap, self._gap_speed(gap, other.speed))
        gap = self._truck_gap(car)
        if gap is not None and gap[1] < 36:
            cap = min(cap, self._gap_speed(gap[0], self.truck.speed))
        return cap

    @staticmethod
    def _gap_speed(gap: float, lead_speed: float) -> float:
        """按车距给目标车速：贴到跟车距离就停，越远越接近前车速度。"""
        safe = CAR_CAR_LEN + 14 + lead_speed * 0.5
        if gap >= safe:
            return 999.0
        if gap <= CAR_CAR_LEN:
            return 0.0
        return max(0.0, lead_speed * (gap - CAR_CAR_LEN) / (safe - CAR_CAR_LEN))

    # ------------------------------------------------------------------ 被行人堵住
    def _update_ped_block(self, dt: float) -> None:
        """消防车被行人顶住：先让行人反方向走开，再重新规划路线继续执行。"""
        t = self.truck
        if t.state not in ("to_fire", "to_refill", "to_repair", "ai_mission") or t.speed > 1:
            self._ped_block_timer = 0.0
            return
        near = [p for p in self.pedestrians
                if math.hypot(p.x - t.x, p.y - t.y) < 70 and self._ped_in_lane(p, 95)]
        if not near:
            self._ped_block_timer = 0.0
            return
        self._ped_block_timer += dt
        if self._ped_block_timer < 2.5:
            return
        self._ped_block_timer = 0.0
        moved = 0
        for ped in near:
            if self._ped_reverse(ped):
                moved += 1
        if self.ai_mission:
            # AI 任务中把「被行人挡住」作为一次感知反馈，重新规划剩余路线
            self._ai_replan("行人堵住车道")
            replanned = True
        else:
            replanned = self._replan_truck()
        if moved or replanned:
            self._log(f"消防车被 {moved} 名行人挡住：行人往反方向走开，"
                      f"消防车重新规划路线继续执行", "warning")

    def _ped_reverse(self, ped: Pedestrian) -> bool:
        """行人原地掉头，沿原路往回走。"""
        back = list(reversed(ped.path[:ped.idx + 1])) if ped.idx < len(ped.path) else []
        if len(back) < 2:
            return False
        ped.path = back
        ped.idx = 1
        ped.waiting = False
        ped.state = "leaving"
        return True

    def _replan_truck(self) -> bool:
        """按当前目标节点重新规划一条路线。"""
        t = self.truck
        if t.goal_node is None:
            return False
        planned = self._plan_route(t.goal_node)
        if planned is None:
            return False
        if planned[0] == t.route:
            return False
        self._set_route(planned[0], t.goal_node)
        return True
    # ------------------------------------------------------------------ 消防车被堵 / 故障
    def _update_clear_way(self, dt: float) -> None:
        """消防车被小车堵住、或在车流里故障：先让小车走，再让消防车回去修。"""
        t = self.truck
        if self.clear_way:
            self.clear_way["timer"] += dt
            if self.clear_way["timer"] >= CLEARWAY_TIME:
                if t.fault and t.state not in ("to_repair", "repairing", "self_check"):
                    self.dispatch_repair()
                    self._log("小车已驶离，消防车返回维修站处理故障", "warning")
                self.clear_way = None
            return

        nearby = any(math.hypot(*[a - b for a, b in zip(self._car_drawn_xy(c),
                                                        (t.x, t.y))]) < CLEARWAY_R
                     for c in self.cars)

        if t.fault and nearby and t.state not in ("to_repair", "repairing"):
            self._start_clear_way("消防车在车流中故障")
            return

        moving_state = t.state in ("to_fire", "to_refill", "to_repair", "ai_mission")
        if moving_state and t.speed < 1.0 and nearby:
            self._truck_stuck += dt
            if self._truck_stuck >= CLEARWAY_TRIGGER:
                self._start_clear_way("消防车被小车堵住")
                return
        else:
            self._truck_stuck = 0.0

    def _start_clear_way(self, reason: str) -> None:
        self.clear_way = {"timer": 0.0, "reason": reason, "started": round(self.time, 1)}
        count = 0
        for car in self.cars:
            x, y = self._car_drawn_xy(car)
            if math.hypot(x - self.truck.x, y - self.truck.y) < CLEARWAY_R:
                car.escape = ESCAPE_TIME
                car.retry = 0.0
                if self._car_flee_from_truck(car):
                    count += 1
        self._log(f"{reason}，{count} 辆小车朝反方向驶离", "warning")

    def _car_flee_from_truck(self, car: NpcCar) -> bool:
        """小车和消防车堵在一起时：朝消防车的反方向开走。"""
        dx, dy = car.x - self.truck.x, car.y - self.truck.y
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return self._car_new_route(car, uturn=True)
        away_x, away_y = dx / d, dy / d            # 背离消防车的单位向量
        nodes = self.city.road.xy
        start = self.city.nearest_road_node(car.x, car.y)
        candidates = []
        for nid, (nx, ny) in enumerate(nodes):
            if nid == start:
                continue
            vx, vy = nx - car.x, ny - car.y
            dist = math.hypot(vx, vy)
            if dist < 60:
                continue
            if (vx * away_x + vy * away_y) / dist < 0.2:
                continue                            # 只走「远离消防车」那一侧
            candidates.append((dist, nid))
        candidates.sort()
        for _dist, goal in candidates[:6]:
            route = graph_astar(self.city.road.adj, start, goal, nodes)
            if not route or len(route) < 2:
                continue
            waypoints = self._lane_waypoints(route)
            if not waypoints:
                continue
            self._apply_car_route(car, route, waypoints, goal)
            return True
        return self._car_new_route(car, uturn=True)  # 兜底：掉头找条路

    # ------------------------------------------------------------------ 路口拥堵疏通
    def _jam_nodes(self) -> list[int]:
        return [nid for nid in range(len(self.city.road.xy))
                if self.city.is_intersection(nid)]

    def _update_jam(self, dt: float) -> None:
        """堵住的路口按「散人 → 小车绕行 → 消防车清理」三步处理。"""
        if self.jam:
            self._advance_jam(dt)
            return
        for node in self._jam_nodes():
            nx, ny = self.city.road.xy[node]
            stuck = 0
            for car in self.cars:
                x, y = self._car_drawn_xy(car)
                if car.speed < 1.0 and math.hypot(x - nx, y - ny) < JAM_R:
                    stuck += 1
            if stuck >= 2:
                self._jam_watch[node] = self._jam_watch.get(node, 0.0) + dt
                if self._jam_watch[node] >= JAM_TRIGGER:
                    self._start_jam(node)
                    return
            else:
                self._jam_watch[node] = 0.0

    def _start_jam(self, node: int) -> None:
        self.jam = {"node": node, "phase": "peds", "timer": 0.0,
                    "started": round(self.time, 1)}
        self.stats["jam_clears"] += 1
        n = self._evacuate_pedestrians(node)
        # 消防车要从当前位置开过去，沿途的行人也先让开，免得堵在半路
        nx, ny = self.city.road.xy[node]
        for ped in self.pedestrians:
            if math.hypot(ped.x - self.truck.x, ped.y - self.truck.y) < 170:
                if self._ped_flee(ped, nx, ny):
                    n += 1
        self._log(f"检测到路口拥堵，先疏散附近行人（{n} 人）", "warning")

    def _advance_jam(self, dt: float) -> None:
        m = self.jam
        m["timer"] += dt
        node = m["node"]
        if m["phase"] == "peds":
            if m["timer"] >= JAM_PED_TIME:
                m["phase"] = "cars"
                m["timer"] = 0.0
                moved = self._evacuate_pedestrians(node)
                aside = self._pull_cars_aside(node)
                self._log(f"行人已自行绕开路口（{moved} 人），{aside} 辆小车靠边让行", "info")
        elif m["phase"] == "cars":
            if m["timer"] >= JAM_CAR_TIME:
                # 不强制消防车过来：只把路让开，消防车需要经过时优先
                self._log("路口已让开，消防车需要经过时优先通过", "success")
                self._end_jam("路口拥堵处理完成，交通恢复")

    def _end_jam(self, reason: str) -> None:
        if self.jam:
            self._jam_watch[self.jam["node"]] = 0.0
        self.jam = None
        self._log(reason, "success")

    def _evacuate_pedestrians(self, node: int) -> int:
        """把路口附近的行人支开：给他们换一条远离路口的路。"""
        nx, ny = self.city.road.xy[node]
        count = 0
        for ped in self.pedestrians:
            if math.hypot(ped.x - nx, ped.y - ny) < JAM_PED_R and self._ped_flee(ped, nx, ny):
                count += 1
        return count

    def _ped_flee(self, ped: Pedestrian, nx: float, ny: float) -> bool:
        # 从「正在走的那条路的节点」接上，不能凭空跳到别处横穿马路
        if ped.idx < len(ped.path):
            start = ped.path[ped.idx]
        else:
            start = self.city.nearest_ped_node(ped.x, ped.y)
        for _attempt in range(10):
            goal = self.city.random_ped_node(self.rng)
            gx, gy = self.city.ped.xy[goal]
            if math.hypot(gx - nx, gy - ny) < 150:
                continue
            path = self.city.ped_route(start, goal)
            if len(path) < 2:
                continue
            # 新路线也不能再穿过这个堵住的路口
            if any(math.hypot(px - nx, py - ny) < 60 for px, py in
                   (self.city.ped.xy[n] for n in path[1:])):
                continue
            ped.path = path
            ped.idx = 1
            ped.waiting = False
            ped.state = "leaving"
            return True
        return False

    def _pull_cars_aside(self, node: int) -> int:
        """让路口附近的小车靠边停下，把车道让给消防车。"""
        nx, ny = self.city.road.xy[node]
        count = 0
        for car in self.cars:
            x, y = self._car_drawn_xy(car)
            if math.hypot(x - nx, y - ny) > 145:
                continue
            car.pull_aside = JAM_PULL_TIME
            count += 1
        return count

    def _reroute_cars_around(self, node: int) -> int:
        """让堵在路口附近的小车掉头、换一条绕开这个路口的路线。"""
        nx, ny = self.city.road.xy[node]
        count = 0
        for car in self.cars:
            x, y = self._car_drawn_xy(car)
            if math.hypot(x - nx, y - ny) > 145:
                continue
            car.retry = 0.0
            if self._car_new_route(car, blocked=(node,), uturn=True):
                count += 1
        return count

    # ------------------------------------------------------------------ 查询
    def _fire_by_id(self, fire_id: int | None) -> Fire | None:
        if fire_id is None:
            return None
        return next((f for f in self.fires if f.id == fire_id), None)

    def get_state(self) -> dict:
        t = self.truck
        spray_target = None
        if t.spray:
            fire = self._fire_by_id(t.target_fire)
            if fire:
                spray_target = {"x": round(fire.x, 1), "y": round(fire.y, 1)}
        arm = None
        if t.arm_target is not None:
            obstacle = next((o for o in self.obstacles if o.id == t.arm_target), None)
            if obstacle:
                arm = {"active": True, "progress": round(t.arm_progress, 2),
                       "target": {"x": round(obstacle.x, 1), "y": round(obstacle.y, 1)}}
        return {
            "time": round(self.time, 1),
            "signal": self.signal,
            "signal_remaining": self.signal_remaining,
            "car_limit": CAR_MAX,
            "truck": {
                "x": round(t.x, 1), "y": round(t.y, 1),
                "heading": round(t.heading, 3),
                "speed": round(t.speed, 1),
                "water": round(t.water, 1),
                "battery": round(t.battery, 1),
                "state": t.state,
                "slowing": t.slowing,
                "fault": t.fault,
                "fault_name": FAULT_NAMES.get(t.fault or "", None),
                "spray": t.spray,
                "spray_target": spray_target,
                "arm": arm,
                "target_fire": t.target_fire,
            },
            "pedestrians": [
                {"id": p.id, "x": round(p.x, 1), "y": round(p.y, 1),
                 "emoji": p.emoji, "state": p.state}
                for p in self.pedestrians
            ],
            "cars": [
                {"id": c.id,
                 # 绘制坐标已含横向让行偏移
                 "x": round(self._car_drawn_xy(c)[0], 1),
                 "y": round(self._car_drawn_xy(c)[1], 1),
                 "heading": round(c.heading, 3),
                 "speed": round(c.speed, 1),
                 "state": c.state,
                 "temporary": c.temporary,
                 "color": c.color}
                for c in self.cars
            ],
            "fires": [
                {"id": f.id, "x": round(f.x, 1), "y": round(f.y, 1),
                 "intensity": round(f.intensity, 1)}
                for f in self.fires
            ],
            "obstacles": [{"id": o.id, "x": round(o.x, 1), "y": round(o.y, 1),
                           "custom": o.custom, "kind": o.kind}
                          for o in self.obstacles],
            "self_check": copy.deepcopy(self.self_check),
            "fov": copy.deepcopy(self.fov),
            "editor": {
                "obstacles": len(self.obstacles),
                "fires": len(self.fires),
                "custom_pedestrians": len([p for p in self.pedestrians if p.custom]),
                "total_pedestrians": len(self.pedestrians),
            },
            "ai": self.ai_status(),
            "jam": ({"node": self.jam["node"], "phase": self.jam["phase"],
                     "x": round(self.city.road.xy[self.jam["node"]][0], 1),
                     "y": round(self.city.road.xy[self.jam["node"]][1], 1)}
                    if self.jam else None),
            "events": self.events[-14:][::-1],
            "stats": dict(self.stats),
        }

    def reset(self) -> None:
        self.__init__(auto_spawn=self.auto_spawn, seed=None)
        self._log("仿真已重置", "info")
