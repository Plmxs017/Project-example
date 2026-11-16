"""地图编辑器后端行为测试。"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim.world import World  # noqa: E402


def make_world(seed=1):
    w = World(auto_spawn=False, seed=seed)
    w.pedestrians = []
    return w


def test_编辑器可以放置路障():
    w = make_world()
    result = w.add_obstacle_at(500, 380)
    assert result["ok"] is True
    assert len(w.obstacles) == 1
    assert w.obstacles[0].custom is True
    assert w.obstacles[0].x == 500.0
    assert w.get_state()["obstacles"][0]["custom"] is True


def test_编辑器新增火情会绑定最近道路并触发自检():
    w = make_world()
    result = w.add_fire_at(500, 300)
    assert result["ok"] is True
    fire = w.fires[0]
    ax, ay = w.city.road.xy[fire.anchor_node]
    assert math.hypot(fire.x - ax, fire.y - ay) <= 80.1
    assert fire.custom is True
    assert w.truck.state == "self_check"
    assert w.self_check["reason"] == "fire"


def test_编辑器新增行人会落在人行道图上():
    w = make_world()
    result = w.add_pedestrian_at(300, 300)
    assert result["ok"] is True
    ped = w.pedestrians[0]
    assert ped.custom is True
    assert len(ped.path) >= 2
    node = w.city.nearest_ped_node(ped.x, ped.y)
    nx, ny = w.city.ped.xy[node]
    assert math.hypot(ped.x - nx, ped.y - ny) < 2.0


def test_编辑器删除最近对象():
    w = make_world()
    w.add_obstacle_at(500, 380)
    w.add_pedestrian_at(300, 300)
    result = w.erase_at(500, 380)
    assert result["ok"] is True
    assert result["kind"] == "obstacle"
    assert w.obstacles == []
    assert len(w.pedestrians) == 1

    result = w.erase_at(300, 300, radius=200)
    assert result["ok"] is True
    assert result["kind"] == "pedestrian"
    assert w.pedestrians == []


def test_编辑器清空动态对象但保留默认行人():
    w = World(auto_spawn=False, seed=2)
    default_count = len(w.pedestrians)
    w.add_obstacle_at(500, 380)
    w.add_fire_at(500, 300)
    w.add_pedestrian_at(300, 300)

    result = w.clear_dynamic()
    assert result["ok"] is True
    assert result["fires"] == 1
    assert result["obstacles"] == 1
    assert result["pedestrians"] == 1
    assert w.fires == []
    assert w.obstacles == []
    assert len(w.pedestrians) == default_count
    assert all(not p.custom for p in w.pedestrians)


def test_编辑器坐标越界会被限制在地图内():
    w = make_world()
    w.add_obstacle_at(-500, 9999)
    obstacle = w.obstacles[0]
    assert 0 < obstacle.x < w.city.WIDTH
    assert 0 < obstacle.y < w.city.HEIGHT
