"""FOV 感知反馈与 AI 路线协同的测试。

覆盖三件事：
1. 不可移开的重障不会被机械臂清理，AI 规划与行驶都会避开它；
2. FOV 把「重障 / 行人占道 / 路口拥堵」整理成反馈，命中 AI 路线时触发重新规划，
   任务不会因为一次重规划而中断；
3. AI 执行途中出现新火情时并入路线，而不是结束任务。
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim.world import (  # noqa: E402
    BASE_SPEED, SLOW_SPEED, Fire, NpcCar, Obstacle, Pedestrian, World,
)


def make_world(seed=1):
    w = World(auto_spawn=False, seed=seed)
    w.pedestrians = []
    return w


def add_fire(w, node, fire_id=1, intensity=60.0):
    """直接放火情，不走 add_fire_at，避免顺带触发全车自检。"""
    x, y = w.city.road.xy[node]
    fire = Fire(id=fire_id, x=x, y=y, anchor_node=node,
                intensity=intensity, checked=True)
    w.fires.append(fire)
    return fire


def advance(w, seconds, step=0.05):
    for _ in range(int(seconds / step)):
        w.update(step)


def test_重障不会被机械臂清理():
    w = make_world(seed=2)
    t = w.truck
    t.state = "to_fire"
    t.x, t.y, t.heading = 154, 300, math.pi / 2
    t.route = [1, 2]
    t.goal_node = 2
    t.waypoints = [(154, 380), (154, 500)]
    t.wp_index = 0
    w.obstacles = [Obstacle(id=9, x=154.0, y=360.0, kind="heavy")]

    advance(w, 2.0)

    assert t.arm_target is None
    assert [o.id for o in w.obstacles] == [9]


def test_AI规划会避开重障所在节点():
    w = make_world(seed=3)
    add_fire(w, 22, 1)
    add_fire(w, 23, 2)
    node = 5                          # (500, 610) 十字路口
    nx, ny = w.city.road.xy[node]
    w.add_obstacle_at(nx, ny, kind="heavy")

    plan = w.plan_ai_route()

    assert plan["ok"] is True
    assert node in plan["blocked_nodes"]
    assert node not in plan["path_nodes"]


def test_只有重障压住目标时该目标会被剔除():
    w = make_world(seed=4)
    fire = add_fire(w, 23, 1)
    nx, ny = w.city.road.xy[fire.anchor_node]
    w.add_obstacle_at(nx, ny, kind="heavy")

    plan = w.plan_ai_route()

    assert plan["ok"] is False
    assert "不可达" in plan["reason"]


def test_FOV发现重障会反馈并让AI重新规划而不是中断任务():
    w = make_world(seed=5)
    add_fire(w, 22, 1)
    add_fire(w, 23, 2)
    add_fire(w, 6, 3)
    assert w.execute_ai_route()["ok"] is True

    advance(w, 8.0)
    t = w.truck
    assert w.ai_mission is not None
    # 在剩余路线的下一个路口放一块移不开的重障
    candidates = [n for n in t.route if n != t.goal_node]
    assert candidates
    node = candidates[0]
    nx, ny = w.city.road.xy[node]
    w.add_obstacle_at(nx, ny, kind="heavy")

    assert w.ai_mission is not None              # 任务没有被中断
    assert w.ai_mission["replans"] >= 1
    assert node not in t.route                   # 新路线避开了重障
    assert any(f["kind"] == "obstacle" for f in w.fov["feedback"])
    assert w.fov["counts"]["obstacle"] >= 1


def test_AI任务中遇到新火情会并入路线():
    w = make_world(seed=6)
    add_fire(w, 22, 1)
    assert w.execute_ai_route()["ok"] is True
    advance(w, 2.0)
    before = len(w.ai_mission["plan"]["stops"])

    w.spawn_fire()

    assert w.ai_mission is not None
    assert w.ai_mission["replans"] >= 1
    assert len(w.ai_mission["plan"]["stops"]) >= before


def test_人行道上的行人不会逼停消防车():
    w = make_world(seed=7)
    t = w.truck
    t.x, t.y, t.heading = 154, 300, math.pi / 2
    ped = Pedestrian(id=1, x=185.0, y=360.0, speed=0.0, emoji="🚶")
    w.pedestrians = [ped]

    assert w._target_speed() == BASE_SPEED
    assert t.slowing is False

    ped.x, ped.y = 154.0, 360.0        # 走到车道里
    speed = w._target_speed()
    assert t.slowing is True
    assert speed <= SLOW_SPEED


def test_行人占道与路口拥堵都会生成FOV反馈():
    w = make_world(seed=8)
    t = w.truck
    t.x, t.y, t.heading = 170, 250, math.pi / 2
    w.pedestrians = [Pedestrian(id=3, x=170.0, y=330.0, speed=0.0, emoji="🚶")]
    w.cars = [
        NpcCar(id=1, x=160.0, y=360.0, heading=math.pi, cruise=60,
               color="#c8102e", speed=0.0),
        NpcCar(id=2, x=182.0, y=364.0, heading=0, cruise=60,
               color="#a80f27", speed=0.0),
    ]

    w._update_fov(0.0)

    kinds = {f["kind"] for f in w.fov["feedback"]}
    assert "pedestrian" in kinds
    assert "jam" in kinds
    assert w.fov["counts"]["pedestrian"] >= 1
    assert w.fov["counts"]["jam"] >= 1
    assert w.fov["blocked_nodes"]


def test_制造拥堵会被FOV识别并在超时后自行疏解():
    w = make_world(seed=10)
    t = w.truck
    t.x, t.y, t.heading = 170, 250, math.pi / 2

    result = w.spawn_jam_demo()
    assert result["ok"] is True
    assert len([c for c in w.cars if c.temporary]) == 2

    w._update_fov(0.0)
    assert w.fov["counts"]["jam"] >= 1
    assert any(f["kind"] == "jam" for f in w.fov["feedback"])

    advance(w, 19.0)
    assert not [c for c in w.cars if c.temporary]


def test_重障压住目标时AI不会谎报完成():
    w = make_world(seed=9)
    fire = add_fire(w, 23, 1)
    nx, ny = w.city.road.xy[fire.anchor_node]
    w.add_obstacle_at(nx, ny, kind="heavy")

    result = w.execute_ai_route()

    assert result["ok"] is False
    assert w.ai_mission is None
    assert len(w.fires) == 1
