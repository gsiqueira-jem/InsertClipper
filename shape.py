import cv2
import numpy as np
from math import cos, sin, pi, radians

class Shape:
    def get_limits(self):
        raise NotImplementedError()
    
    def draw(self, img, offset, scale):
        raise NotImplementedError()
    
    @staticmethod
    def normalize(point, offset, scale):
        return int((point[0] - offset[0]) * scale),  int((point[1] - offset[1]) * scale)


class Line(Shape):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def get_limits(self):
        xs = [self.start[0], self.end[0]]
        ys = [self.start[1], self.end[1]]
        
        return min(xs), max(xs), min(ys), max(ys)
    
    def draw(self, img, off, sc):
        p0 = Shape.normalize(self.start, off, sc)
        p1 = Shape.normalize(self.end, off, sc)
        cv2.line(img, p0, p1, (0, 0, 0), 1)

class Circle(Shape):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_limits(self):
        cx, cy, _ = self.center
        r = self.radius
        return cx - r, cx + r, cy - r, cy + r

    def draw(self, img, off, sc):
        center_norm = Shape.normalize(self.center, off, sc)
        radius_norm = int(self.radius * sc)
        cv2.circle(img, center_norm, radius_norm, (0, 0, 0), 1)


class Arc(Shape):
    def __init__(self, center, radius, start_angle, end_angle):
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle

    def get_limits(self):
        # Sample a few points along the arc

        angles = [self.start_angle, self.end_angle]
        if self.start_angle < self.end_angle:
            angles += [a for a in [0, 90, 180, 270] if self.start_angle < a < self.end_angle]
        else:  # wrapped
            angles += [0, 90, 180, 270]

        points = [
            (
                self.center[0] + self.radius * cos(radians(a)),
                self.center[1] + self.radius * sin(radians(a)),
            )
            for a in angles
        ]
        xs, ys = zip(*points)
        return min(xs), max(xs), min(ys), max(ys)

    def draw(self, img, offset, scale):
        center_px = Shape.normalize(self.center, offset, scale)
        radius_px = int(self.radius * scale)
        cv2.ellipse(img, center_px, (radius_px, radius_px), 0, self.start_angle, self.end_angle, (0, 0, 0), 1)

class Polyline(Shape):
    def __init__(self, points, closed):
        if closed and points[0] != points[-1]:
            points = points + [points[0]]
        
        self.lines = [Line(points[i], points[i+1]) for i in range(len(points) - 1)]
    
    def get_limits(self):
        if not self.lines:
            return 0, 0, 0, 0 
        
        min_xs, max_xs, min_ys, max_ys = zip(*(line.get_limits() for line in self.lines))
        return min(min_xs), max(max_xs), min(min_ys), max(max_ys)
    
    def draw(self, img, off, sc):
        for line in self.lines:
            line.draw(img, off, sc) 

class Ellipse(Shape):
    def __init__(self, center, major_axis, ratio, start_param, end_param):
        self.center = center
        self.major_axis = major_axis
        self.ratio = ratio
        self.start_param = start_param
        self.end_param = end_param

    def get_limits(self):
        # Approximate the ellipse arc using point sampling
        steps = 20
        points = []
        for i in range(steps + 1):
            t = self.start_param + (self.end_param - self.start_param) * i / steps
            x = self.major_axis[0] * cos(t)
            y = self.major_axis[1] * cos(t)  # actually unused, see below
            px = self.major_axis[0] * cos(t)
            py = self.major_axis[0] * self.ratio * sin(t)
            points.append((self.center[0] + px, self.center[1] + py))

        xs, ys = zip(*points)
        return min(xs), max(xs), min(ys), max(ys)

    def draw(self, img, offset, scale):
        # OpenCV doesn't directly support DXF-style rotated ellipses, so we approximate with full ellipse
        major_length = (self.major_axis[0] ** 2 + self.major_axis[1] ** 2) ** 0.5
        angle = np.degrees(np.arctan2(self.major_axis[1], self.major_axis[0]))

        center_px = Shape.normalize(self.center, offset, scale)
        axes_px = (int(major_length * scale), int(major_length * self.ratio * scale))

        cv2.ellipse(
            img, center_px, axes_px, angle,
            np.degrees(self.start_param), np.degrees(self.end_param),
            (0, 0, 0), 1
        )




def entity2shape(e):
    dxf_type = e.dxftype()

    if dxf_type == "LINE":
        return Line(e.dxf.start, e.dxf.end)
    elif dxf_type == "CIRCLE":
        return Circle(e.dxf.center, e.dxf.radius)
    elif dxf_type == "ARC":
        return Arc(e.dxf.center, e.dxf.radius, e.dxf.start_angle, e.dxf.end_angle)
    elif dxf_type == "ELLIPSE":
        return Ellipse(e.dxf.center, e.dxf.major_axis, e.dxf.ratio, e.dxf.start_param, e.dxf.end_param)
    elif dxf_type == "LWPOLYLINE":
        return Polyline(e.get_points(), e.closed)
    elif dxf_type == "POLYLINE":
        coords = [v.dxf.location for v in e.vertices]
        return Polyline(coords, e.is_closed)
    else:
        return None