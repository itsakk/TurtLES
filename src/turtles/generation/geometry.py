"""Gmsh geometry (``.geo``) of a rotated super-ellipse in the TurtLES domain.

Layout (paper, Appendix F): four structured boundary-layer blocks around the
object, an unstructured wake box with refinement, a far-field domain closed by an
inlet arc and an outlet at ``x = 25``, extruded over ``z ∈ [0, 2π]`` in 30 layers
with periodic side walls. Cell sizes are interpolated by a power law between
reference meshes at Re = 1000 and Re = 3900.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Reference mesh sizes used for the power-law interpolation across Reynolds numbers.
_REF_LOW = {"Re": 1000.0, "l_theta": math.pi / 50, "l_t": 0.051, "l_wb": 0.287, "l_ff": 0.9}
_REF_HIGH = {"Re": 3900.0, "l_theta": math.pi / 116, "l_t": 0.022, "l_wb": 0.22, "l_ff": 0.868}

SPANWISE_LENGTH = 2 * math.pi
SPANWISE_LAYERS = 30
FARFIELD_RADIUS = 9.0
OUTLET_X = 25.0


def power_law_interpolate(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
    """Evaluate the power law through ``(x1, y1)`` and ``(x2, y2)`` at ``x``."""
    exponent = math.log(y2 / y1) / math.log(x2 / x1)
    return y1 / (x1**exponent) * x**exponent


def cell_sizes(Re: float) -> dict[str, float]:
    """Wall-tangential, first-cell, wake-box and far-field cell sizes at ``Re``."""
    return {
        key: power_law_interpolate(
            _REF_LOW["Re"], _REF_LOW[key], _REF_HIGH["Re"], _REF_HIGH[key], Re
        )
        for key in ("l_theta", "l_t", "l_wb", "l_ff")
    }


def boundary_layer_cells(thickness: float, first_height: float, growth: float) -> int:
    """Number of geometrically growing cells spanning ``thickness``."""
    return math.ceil(math.log(1 + thickness * (growth - 1) / first_height) / math.log(growth))


def superellipse_extremes(a: float, b: float, n: float, theta: float, resolution: int = 1000):
    """Leftmost, rightmost, topmost and bottommost points of the rotated super-ellipse."""
    points = []
    for i in range(resolution):
        t = 2 * math.pi * i / resolution
        cx, sy = math.cos(t), math.sin(t)
        x = a * abs(cx) ** (2 / n) * (1 if cx >= 0 else -1)
        y = b * abs(sy) ** (2 / n) * (1 if sy >= 0 else -1)
        points.append(
            (x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta))
        )
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (
        points[int(np.argmin(xs))],
        points[int(np.argmax(xs))],
        points[int(np.argmax(ys))],
        points[int(np.argmin(ys))],
    )


def _quadrant_angles(a: float, b: float, n: float, quadrant: int) -> tuple[float, float]:
    if n >= 5:
        offsets_y = [b, b, -b, -b]
        offsets_x = [a, -a, -a, a]
        start = math.atan2(offsets_y[quadrant - 1], offsets_x[quadrant - 1]) % (2 * math.pi)
        end = math.atan2(offsets_y[quadrant % 4], offsets_x[quadrant % 4]) % (2 * math.pi)
        if end < start:
            end += 2 * math.pi
        return start, end - start
    return (quadrant - 1) * math.pi / 2, math.pi / 2


def _quadrant(a, b, n, N, p_start, p_end, spline_id, quadrant, angle0, span, angle) -> list[str]:
    if math.isinf(n):
        corners_start = [(a, b), (-a, b), (-a, -b), (a, -b)]
        corners_end = [(-a, b), (-a, -b), (a, -b), (a, b)]
        c, s = math.cos(angle), math.sin(angle)
        rot = lambda p: (p[0] * c - p[1] * s, p[0] * s + p[1] * c)  # noqa: E731
        q = quadrant - 1
        start = rot(corners_start[q])
        end = rot(corners_end[q])
        lines = ["", "// Rectangle edge"]
        if q == 0:
            lines.append(f"Point({p_start}) = {{{start[0]}, {start[1]}, 0}};")
        if q < 3:
            lines.append(f"Point({p_end}) = {{{end[0]}, {end[1]}, 0}};")
        lines.append(f"Line({spline_id}) = {{{p_start}, {p_end}}};")
        return lines
    loops = [
        f"For i In {{0 : {N}}}",
        f"For i In {{1 : {N}}}",
        f"For i In {{1 : {N}}}",
        f"For i In {{1 : {N - 1}}}",
    ]
    return [
        "",
        "// Super-ellipse quadrant",
        loops[quadrant - 1],
        f"    t = {angle0} + {span} * i / {N};",
        "    cx = Cos(t);",
        "    sy = Sin(t);",
        f"    x = {a} * (Fabs(cx))^(2/{n}) * (cx < 0 ? -1 : 1);",
        f"    y = {b} * (Fabs(sy))^(2/{n}) * (sy < 0 ? -1 : 1);",
        "    x_rot = x * Cos(inflow_angle) - y * Sin(inflow_angle);",
        "    y_rot = x * Sin(inflow_angle) + y * Cos(inflow_angle);",
        f"    Point(i + {p_start}) = {{x_rot, y_rot, 0}};",
        "EndFor",
        f"Spline({spline_id}) = {{{p_start} : {p_start + N - 1}, {p_end}}};",
    ]


def _lame_curve(a, b, n, N, p_start, p_ends, spline_ids, loop_id, angle) -> list[str]:
    lines: list[str] = []
    for j in range(4):
        angle0, span = _quadrant_angles(a, b, n, j + 1)
        lines += _quadrant(
            a, b, n, N, p_start + j * N, p_ends[j], spline_ids[j], j + 1, angle0, span, angle
        )
    lines += ["", f"Curve Loop({loop_id}) = {{{', '.join(map(str, spline_ids))}}};"]
    return lines


def _connect_object_to_bl(
    obj_points, bl_points, obj_splines, bl_splines, line_ids, block_ids
) -> list[str]:
    lines = ["", "// Structured boundary-layer blocks"]
    lines += [f"Line({line_ids[i]}) = {{{obj_points[i]}, {bl_points[i]}}};" for i in range(4)]
    for i in range(4):
        j = (i + 1) % 4
        lines += [
            f"Curve Loop({block_ids[i]}) = {{{obj_splines[i]}, {line_ids[j]}, -{bl_splines[i]}, -{line_ids[i]}}};",
            f"Plane Surface({block_ids[i]}) = {{{block_ids[i]}}};",
        ]
    ids = ", ".join(map(str, line_ids))
    blocks = ", ".join(map(str, block_ids))
    lines += [
        f"Transfinite Curve {{{ids}}} = n_BL Using Progression GR_BL;",
        f"Transfinite Curve {{{obj_splines[0]}, {obj_splines[2]}, {bl_splines[0]}, {bl_splines[2]}}} = n_longside;",
        f"Transfinite Curve {{{obj_splines[1]}, {obj_splines[3]}, {bl_splines[1]}, {bl_splines[3]}}} = n_shortside;",
        f"Transfinite Surface {{{blocks}}};",
        f"Recombine Surface {{{blocks}}};",
    ]
    return lines


def generate_geo(
    *,
    Re: float,
    inflow_angle: float,
    a: float,
    b: float,
    n: float,
    growth_rate: float = 1.021,
    bl_thickness: float = 0.5,
    curve_points: int = 100,
) -> str:
    """Return the ``.geo`` source for one case (``inflow_angle`` in radians)."""
    sizes = cell_sizes(Re)
    if n == 1:
        long_side = short_side = math.sqrt(a**2 + b**2)
    elif n == 2:  # Ramanujan's quarter-perimeter approximation
        long_side = short_side = math.pi / 4 * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    else:
        long_side, short_side = 2 * a, 2 * b
    n_bl = boundary_layer_cells(bl_thickness, sizes["l_t"], growth_rate)

    lines = [
        "// Generated by turtles.generation.geometry",
        'SetFactory("OpenCASCADE");',
        "",
        f"z_len = {SPANWISE_LENGTH};",
        f"n_z   = {SPANWISE_LAYERS};",
        f"lc_wall = {sizes['l_theta']};",
        f"t_first = {sizes['l_t']};",
        f"lc_wake = {sizes['l_wb']};",
        f"lc_ff   = {sizes['l_ff']};",
        f"GR_BL   = {growth_rate};",
        f"t_BL    = {bl_thickness};",
        f"n_BL    = {n_bl};",
        f"n_longside  = {long_side / sizes['l_theta']};",
        f"n_shortside = {short_side / sizes['l_theta']};",
        f"inflow_angle = {inflow_angle};",
    ]

    N = curve_points
    obj_splines = [1, 2, 3, 4]
    p0 = 1000
    obj_ends = [p0 + N, p0 + 2 * N, p0 + 3 * N, p0]
    lines += _lame_curve(a, b, n, N, p0, obj_ends, obj_splines, 10, inflow_angle)

    a_bl, b_bl = a + bl_thickness, b + bl_thickness
    n_bl_shape = {1: 1.5, 2: 2, 5: 5}.get(n, 10 if math.isinf(n) else n)
    bl_splines = [11, 12, 13, 14]
    p0_bl = 4000
    bl_ends = [p0_bl + N, p0_bl + 2 * N, p0_bl + 3 * N, p0_bl]
    lines += _lame_curve(a_bl, b_bl, n_bl_shape, N, p0_bl, bl_ends, bl_splines, 20, inflow_angle)

    obj_points = [obj_ends[-1], *obj_ends[:-1]]
    bl_points = [bl_ends[-1], *bl_ends[:-1]]
    block_ids = [41, 42, 43, 44]
    lines += _connect_object_to_bl(
        obj_points, bl_points, obj_splines, bl_splines, [31, 32, 33, 34], block_ids
    )

    left, right, top, bottom = superellipse_extremes(a_bl, b_bl, n_bl_shape, inflow_angle)
    r_wb, l_wb, t_wb, b_wb = right[0] + 7, left[0] - 1, top[1] + 1, bottom[1] - 1
    wb = [51, 52, 53, 54]
    lines += [
        "",
        "// Wake refinement box",
        f"Point({wb[0]}) = {{{r_wb}, {t_wb}, 0}};",
        f"Point({wb[1]}) = {{{l_wb}, {t_wb}, 0}};",
        f"Point({wb[2]}) = {{{l_wb}, {b_wb}, 0}};",
        f"Point({wb[3]}) = {{{r_wb}, {b_wb}, 0}};",
        f"Line({wb[0]}) = {{{wb[0]}, {wb[1]}}};",
        f"Line({wb[1]}) = {{{wb[1]}, {wb[2]}}};",
        f"Line({wb[2]}) = {{{wb[2]}, {wb[3]}}};",
        f"Line({wb[3]}) = {{{wb[3]}, {wb[0]}}};",
        f"Curve Loop(60) = {{{', '.join(map(str, wb))}}};",
    ]

    shift = l_wb + 2
    d = [61, 62, 63, 64, 65, 66]
    lines += [
        "",
        "// Far field",
        f"Point({d[0]}) = {{{shift}, {FARFIELD_RADIUS}, 0}};",
        f"Point({d[1]}) = {{{-FARFIELD_RADIUS + shift}, 0, 0}};",
        f"Point({d[2]}) = {{{shift}, {-FARFIELD_RADIUS}, 0}};",
        f"Point({d[3]}) = {{{OUTLET_X}, {-FARFIELD_RADIUS}, 0}};",
        f"Point({d[4]}) = {{{OUTLET_X}, {FARFIELD_RADIUS}, 0}};",
        f"Point({d[5]}) = {{{shift}, 0, 0}};",
        f"Circle({d[0]}) = {{{d[0]}, {d[5]}, {d[1]}}};",
        f"Circle({d[1]}) = {{{d[1]}, {d[5]}, {d[2]}}};",
        f"Line({d[2]}) = {{{d[2]}, {d[3]}}};",
        f"Line({d[3]}) = {{{d[3]}, {d[4]}}};",
        f"Line({d[4]}) = {{{d[4]}, {d[0]}}};",
        f"Curve Loop(70) = {{{d[0]}, {d[1]}, {d[2]}, {d[3]}, {d[4]}}};",
        "",
        "Plane Surface(101) = {60, 20};",
        "Plane Surface(201) = {70, 60};",
    ]

    surfaces = ", ".join(map(str, [101, 201, *block_ids]))
    lines += [
        "",
        "// Extrusion and periodic side walls",
        "out[] = Extrude {0, 0, z_len} {",
        f"  Surface{{{surfaces}}};",
        "  Layers{n_z};",
        "  Recombine;",
        "};",
        "Periodic Surface {out[0]}  = {101} Translate {0,0,z_len};",
        "Periodic Surface {out[10]} = {201} Translate {0,0,z_len};",
        f"Periodic Surface {{out[21]}} = {{{block_ids[0]}}} Translate {{0,0,z_len}};",
        f"Periodic Surface {{out[27]}} = {{{block_ids[1]}}} Translate {{0,0,z_len}};",
        f"Periodic Surface {{out[33]}} = {{{block_ids[2]}}} Translate {{0,0,z_len}};",
        f"Periodic Surface {{out[39]}} = {{{block_ids[3]}}} Translate {{0,0,z_len}};",
        'Physical Volume("Fluid") = {out[1], out[11], out[22], out[28], out[34], out[40]};',
        'Physical Surface("inlet")  = {out[12], out[13]};',
        'Physical Surface("outlet") = {out[14], out[15], out[16]};',
        'Physical Surface("wall")   = {out[23], out[29], out[35], out[41]};',
        f'Physical Surface("periodic-0-l") = {{{surfaces}}};',
        'Physical Surface("periodic-0-r") = {out[0], out[10], out[21], out[27], out[33], out[39]};',
        "",
        "// Size fields: wake box and wall refinement",
        "Field[1] = Distance;",
        f"Field[1].CurvesList = {{{', '.join(map(str, wb))}}};",
        "Field[1].Sampling = 100;",
        "Field[2] = Threshold;",
        "Field[2].InField = 1;",
        "Field[2].SizeMin = lc_wake;",
        "Field[2].SizeMax = lc_ff;",
        f"Field[2].DistMin = {t_wb} + 0.01;",
        f"Field[2].DistMax = {t_wb} + 0.02;",
        "Field[3] = Restrict;",
        "Field[3].InField = 2;",
        "Field[3].SurfacesList = {101};",
        "Field[3].VolumesList = {out[1]};",
        "Field[4] = Distance;",
        f"Field[4].CurvesList = {{{', '.join(map(str, obj_splines))}}};",
        "Field[4].Sampling = 100;",
        "Field[5] = Threshold;",
        "Field[5].InField = 4;",
        "Field[5].SizeMin = lc_wall;",
        "Field[5].SizeMax = lc_ff;",
        "Field[5].DistMin = t_BL;",
        "Field[5].DistMax = t_BL + 0.5;",
        "Field[7] = Min;",
        "Field[7].FieldsList = {3, 5};",
        "Background Field = 7;",
        "Mesh.Algorithm = 6;",
        "Mesh.ElementOrder = 4;",
        "Mesh.HighOrderOptimize = 0;",
    ]
    return "\n".join(lines) + "\n"


def write_geo(path: str | Path, **kwargs) -> Path:
    """Write :func:`generate_geo` output to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_geo(**kwargs))
    return path
