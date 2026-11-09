"""main.py —— 自动消防小车仿真后端

启动：cd backend && python -m uvicorn main:app --reload
页面：http://127.0.0.1:8000
接口：http://127.0.0.1:8000/docs

职责划分：
- sim/city.py   城市路网、人行道、站点、火情点位（静态底图）
- sim/world.py  车辆状态机、行人、火情、机械臂、自检（仿真规则）
- ai/detector.py 告警规则
- 本文件        只做 HTTP 接口、后台心跳线程和静态文件托管
"""
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ai.detector import check
from sim.world import World

app = FastAPI(title="自动消防小车仿真系统")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

world = World()
world_lock = threading.Lock()


def game_loop() -> None:
    """20Hz 心跳：唯一推进仿真时间的地方，HTTP 接口只读写状态。"""
    last = time.monotonic()
    while True:
        now = time.monotonic()
        dt = now - last
        last = now
        with world_lock:
            world.update(dt)
        time.sleep(0.05)


threading.Thread(target=game_loop, daemon=True).start()


def _snapshot() -> dict:
    state = world.get_state()
    state["alerts"] = check(state)
    return state


@app.get("/api/city")
def api_city():
    """静态地图：道路、建筑、树木、站点、火情点位，前端只取一次。"""
    with world_lock:
        return world.city.to_dict()


@app.get("/api/state")
def api_state():
    """一帧完整状态：车辆、行人、火情、路障、自检、事件、告警。"""
    with world_lock:
        return _snapshot()


@app.get("/api/alerts")
def api_alerts():
    with world_lock:
        return {"alerts": check(world.get_state())}


@app.post("/api/selfcheck")
def api_selfcheck():
    """手动触发一次全车自检。"""
    with world_lock:
        return world.start_self_check("manual")


@app.post("/api/fire")
def api_fire():
    """模拟传感器检测到火情（正式流程会自动触发自检）。"""
    with world_lock:
        return world.spawn_fire()


@app.post("/api/fault")
def api_fault(kind: str | None = None):
    """注入功能故障，可选 kind=arm/pump/sensor/nav。"""
    with world_lock:
        return world.inject_fault(kind)


@app.post("/api/obstacle")
def api_obstacle(ahead: bool = False):
    """放置路障；ahead=true 时放在小车正前方，方便演示机械臂清障。"""
    with world_lock:
        return world.spawn_obstacle(ahead=ahead)


@app.post("/api/reset")
def api_reset():
    global world
    with world_lock:
        world = World()
    return {"ok": True}


@app.post("/api/ai/route")
def api_ai_route():
    """AI 最优路线：A* 求两两最短路 + DP / 2-opt 求访问顺序，只规划不执行。"""
    with world_lock:
        return world.plan_ai_route()


@app.post("/api/ai/execute")
def api_ai_execute():
    """AI 接管：车辆按规划好的顺序依次前往各目标点。"""
    with world_lock:
        return world.execute_ai_route()


@app.post("/api/ai/cancel")
def api_ai_cancel():
    with world_lock:
        return world.cancel_ai()


# ------------------------------------------------------------------ FOV / 地图编辑器
@app.post("/api/fov/toggle")
def api_fov_toggle(enabled: bool | None = None):
    """开启 / 关闭 FOV 感知寻路；不传 enabled 时切换。"""
    with world_lock:
        return world.set_fov(enabled)


@app.post("/api/editor/obstacle")
def api_editor_obstacle(x: float, y: float):
    """地图编辑器：在画布坐标 (x, y) 放置路障。"""
    with world_lock:
        return world.add_obstacle_at(x, y)


@app.post("/api/editor/fire")
def api_editor_fire(x: float, y: float):
    """地图编辑器：在画布坐标 (x, y) 新增火情。"""
    with world_lock:
        return world.add_fire_at(x, y)


@app.post("/api/editor/pedestrian")
def api_editor_pedestrian(x: float, y: float):
    """地图编辑器：在最近的人行道节点新增行人。"""
    with world_lock:
        return world.add_pedestrian_at(x, y)


@app.post("/api/editor/erase")
def api_editor_erase(x: float, y: float):
    """地图编辑器：删除点击位置附近的动态对象。"""
    with world_lock:
        return world.erase_at(x, y)


@app.post("/api/editor/clear")
def api_editor_clear():
    """地图编辑器：清空全部火情、路障和自定义行人。"""
    with world_lock:
        return world.clear_dynamic()


app.mount("/", StaticFiles(directory="static", html=True), name="static")
