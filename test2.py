from dataclasses import dataclass

@dataclass
class Point:
    x : float
    y : float

class test(Point):
    def __init__(self) -> None:
        Point.__init__(self)

    def create():
        point = Point(1,2)
        return point

a = test()
print(a.create())
