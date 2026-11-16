"""FOV 视野与可见性寻路测试。"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from algo.fov import (  # noqa: E402
    fov_astar, line_of_sight, point_in_fov, visibility_polygon,
)
from sim.world import Obstacle, World  # noqa: E402


def test_建筑会挡住视线():
    buildings = [{"x": 80, "y": 0, "w": 40, "h": 200}]
    assert line_of_sight((0, 100), (60, 100), buildings) is True
    assert line_of_sight((0, 100), (160, 100), buildings) is False


def test_视野锥只覆盖车头方向():
    assert point_in_fov((0, 0), 0, math.radians(45), 100, (50, 0)) is True
    assert point_in_fov((0, 0), 0, math.radians(45), 100, (-50, 0)) is False
    # 侧后方不在 90° 视野锥内
    assert point_in_fov((0, 0), 0, math.radians(45), 100, (0, 50)) is False


def test_视野多边形会被建筑切短():
    buildings = [{"x": 60, "y": -50, "w": 20, "h": 100}]
    poly = visibility_polygon((0, 0), 0, math.radians(45), 120, buildings, rays=12)
    assert len(poly) == 14
    assert poly[0] == (0.0, 0.0)
    # 正前方射线到建筑面上就停住，不会画到 120
    center = poly[len(poly) // 2]
    assert math.hypot(center[0], center[1]) < 120


def test_FOV寻路优先选择视野内的稍长路线():
    adj = {
        0: [(1, 10.0), (3, 10.5)],
        1: [(0, 10.0), (2, 10.0)],
        2: [(1, 10.0), (5, 10.5), (3, 10.5)],
        3: [(0, 10.5), (4, 10.5), (2, 10.5)],
        4: [(3, 10.5), (5, 10.5)],
        5: [(4, 10.5), (2, 10.5)],
    }
    # 0-1-2 更短但完全在视野外；0-3-2 稍长但可见。
    route = fov_astar(adj, 0, 2, None, visible={0, 3, 4, 5, 2},
                      invisible_penalty=1.28)
    assert route is not None
    assert 1 not in route
    assert route[0] == 0 and route[-1] == 2


def test_FOV寻路可以避开被遮挡的节点():
    coords = [(0, 0), (10, 0), (20, 0), (10, 10)]
    adj = {
        0: [(1, 10.0), (3, 10.0)],
        1: [(0, 10.0), (2, 10.0)],
        2: [(1, 10.0), (3, 10.0)],
        3: [(0, 10.0), (2, 10.0)],
    }
    route = fov_astar(adj, 0, 2, coords, blocked={1})
    assert route == [0, 3, 2]


def test_世界状态包含FOV观测与遮挡轮廓():
    w = World(auto_spawn=False, seed=1)
    w.pedestrians = []
    w.truck.x, w.truck.y, w.truck.heading = 154, 300, math.pi / 2
    w.obstacles = [Obstacle(id=77, x=170, y=500, custom=True)]
    w._update_fov()

    fov = w.get_state()["fov"]
    assert fov["enabled"] is True
    assert fov["half_angle_deg"] == 55.0
    assert len(fov["polygon"]) > 10
    assert any(e["kind"] == "obstacle" and e["id"] == 77 for e in fov["entities"])


def test_FOV发现占道路障后会重新规划():
    w = World(auto_spawn=False, seed=1)
    w.pedestrians = []
    t = w.truck
    t.state = "to_fire"
    t.x, t.y, t.heading = 154, 300, math.pi / 2
    t.route = [1, 2, 22, 5]            # 原路线经过被堵的 2 号节点
    t.goal_node = 5
    t.waypoints = [(170, 380), (170, 610), (300, 610), (500, 610)]
    t.wp_index = 0
    w.obstacles = [Obstacle(id=88, x=170, y=500, custom=True)]

    w._update_fov()

    assert w.fov["replans"] >= 1
    assert w.truck.route != [1, 2, 22, 5]
    assert 2 not in w.truck.route
    assert any("FOV" in e["text"] and "重新规划" in e["text"] for e in w.events)


def test_FOV开关能恢复关闭状态():
    w = World(auto_spawn=False, seed=1)
    assert w.get_state()["fov"]["enabled"] is True
    assert w.set_fov(False)["enabled"] is False
    assert w.get_state()["fov"]["enabled"] is False
    assert w.set_fov()["enabled"] is True
