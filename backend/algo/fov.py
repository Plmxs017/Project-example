"""fov.py —— 视野（FOV）感知与可见性寻路

这个模块只做几何和算法，不依赖 World，方便单独测试：
- 视野锥：以车辆位置为顶点、车头方向为中心，按角度采样射线；
- 视线遮挡：建筑矩形会挡住射线，形成真实的扇形视野；
- 可见性 A*：在加权路网上，给「视野外」的边增加探索代价，让车辆优先沿着
  当前能看见 / 已经探索过的路走；同时可以避开 FOV 发现的路障节点。

坐标与项目其它模块一致：画布左上角为 (0, 0)，x 向右、y 向下。
"""
from __future__ import annotations

import math
from heapq import heappop, heappush
from typing import Iterable

Point = tuple[float, float]
Rect = dict


def angle_diff(a: float, b: float) -> float:
    """返回 a、b 两个角度之间的最短夹角（-π..π）。"""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def point_in_fov(origin: Point, heading: float, half_angle: float,
                 radius: float, point: Point) -> bool:
    """点是否落在视野扇形内（不考虑建筑遮挡）。"""
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    dist = math.hypot(dx, dy)
    if dist <= 1e-6 or dist > radius:
        return False
    return abs(angle_diff(math.atan2(dy, dx), heading)) <= half_angle


def _ray_rect_distance(px: float, py: float, dx: float, dy: float,
                       rect: Rect, limit: float) -> float | None:
    """射线与轴对齐建筑矩形最近交点的距离；没有交点返回 None。"""
    x0, y0 = float(rect["x"]), float(rect["y"])
    x1, y1 = x0 + float(rect["w"]), y0 + float(rect["h"])
    tmin, tmax = -math.inf, math.inf

    for p, d, lo, hi in ((px, dx, x0, x1), (py, dy, y0, y1)):
        if abs(d) < 1e-9:
            if p < lo or p > hi:
                return None
            continue
        t1, t2 = (lo - p) / d, (hi - p) / d
        if t1 > t2:
            t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return None

    if tmax < 0:
        return None
    # 从矩形内部出发时，tmin 可能小于 0；视野射线仍应从当前位置开始。
    hit = max(0.0, tmin)
    if hit > limit:
        return None
    return hit


def line_of_sight(a: Point, b: Point, buildings: Iterable[Rect],
                  margin: float = 0.0) -> bool:
    """两点之间是否有建筑遮挡（忽略端点所在建筑）。"""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return True
    dx, dy = dx / length, dy / length
    for rect in buildings:
        # 给建筑留一点内缩，避免把贴着建筑门口的实体误判为被遮挡。
        shrunk = {
            "x": float(rect["x"]) + margin,
            "y": float(rect["y"]) + margin,
            "w": max(0.0, float(rect["w"]) - 2 * margin),
            "h": max(0.0, float(rect["h"]) - 2 * margin),
        }
        hit = _ray_rect_distance(ax, ay, dx, dy, shrunk, length)
        if hit is not None and 0.5 < hit < length - 0.5:
            return False
    return True


def visibility_polygon(origin: Point, heading: float, half_angle: float,
                       radius: float, buildings: Iterable[Rect],
                       rays: int = 41) -> list[Point]:
    """计算视野锥的多边形轮廓：origin + 每条射线的命中点。

    前端直接把这个多边形填成半透明扇形；没有建筑遮挡时就是标准圆弧。
    """
    ox, oy = origin
    rays = max(3, int(rays))
    polygon: list[Point] = [(round(ox, 1), round(oy, 1))]
    for i in range(rays + 1):
        angle = heading - half_angle + (2 * half_angle) * i / rays
        dx, dy = math.cos(angle), math.sin(angle)
        dist = radius
        for rect in buildings:
            hit = _ray_rect_distance(ox, oy, dx, dy, rect, radius)
            if hit is not None and hit < dist:
                dist = hit
        dist = max(2.0, dist)
        polygon.append((round(ox + dx * dist, 1), round(oy + dy * dist, 1)))
    return polygon


def visible_nodes(coords: Iterable[Point], origin: Point, heading: float,
                  half_angle: float, radius: float,
                  buildings: Iterable[Rect]) -> set[int]:
    """返回当前视野内、且没有被建筑挡住的路网节点编号。"""
    buildings = list(buildings)
    seen: set[int] = set()
    for nid, point in enumerate(coords):
        if not point_in_fov(origin, heading, half_angle, radius, point):
            continue
        if line_of_sight(origin, point, buildings, margin=1.0):
            seen.add(nid)
    return seen


def fov_astar(adj: dict[int, list[tuple[int, float]]], start: int, goal: int,
              coords: list[Point] | tuple[Point, ...] | None = None,
              visible: set[int] | None = None,
              blocked: Iterable[int] = (),
              invisible_penalty: float = 1.28) -> list[int] | None:
    """带 FOV 探索代价的 A*。

    - 终点始终可达；
    - 视野外 / 尚未探索的节点仍然可走，但每条边乘以 ``invisible_penalty``；
    - ``blocked`` 中的节点直接禁止通过，用于绕开 FOV 发现的路障。
    """
    blocked_set = set(blocked)
    blocked_set.discard(goal)
    if start == goal:
        return [start]
    if start in blocked_set or goal in blocked_set:
        return None

    visible_set = visible or set()

    def h(nid: int) -> float:
        if coords is None:
            return 0.0
        x, y = coords[nid]
        gx, gy = coords[goal]
        return math.hypot(x - gx, y - gy)

    open_heap = [(h(start), 0.0, start)]
    g = {start: 0.0}
    parent: dict[int, int | None] = {start: None}
    closed: set[int] = set()
    while open_heap:
        _, cur_g, cur = heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path: list[int] = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        closed.add(cur)
        for nxt, cost in adj.get(cur, ()):
            if nxt in blocked_set or nxt in closed:
                continue
            explore_cost = cost * (1.0 if nxt in visible_set else invisible_penalty)
            ng = cur_g + explore_cost
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                parent[nxt] = cur
                heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None
