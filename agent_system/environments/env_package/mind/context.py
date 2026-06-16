from abc import ABC
from copy import deepcopy


class ObjectBase(ABC):
    def __init__(self, name: str, position: tuple[int, int], direction: str | None = None):
        self.name = name
        self._position = position
        self._direction = direction
        self.environment = None
    
    def __repr__(self) -> str:
        return f"ObjectBase(name={self.name}, position={self._position}, direction={self._direction})"

    def __eq__(self, other) -> bool:
        """判断两个对象是否相等，基于 name、position 和 direction"""
        if not isinstance(other, ObjectBase):
            return False
        return (self.name == other.name and
                self._position == other._position and
                self._direction == other._direction)
    
    def __hash__(self) -> int:
        """使对象可以作为字典键或集合元素"""
        return hash((self.name, self._position, self._direction))

    def position(self) -> tuple[int, int]:
        return self._position

    def direction(self) -> str:
        return self._direction

    def setup_environment(self, environment: list['ObjectBase']) -> None:
        self.environment = environment

    def front(self, view: 'ObjectBase') -> list['ObjectBase']:
        assert self.environment is not None, "Environment is not set, please call setup_environment first"
        ret = [deepcopy(obj) for obj in self.environment if ObjectBase.is_front(view, self, obj)]

        return ret

    def right(self, view: 'ObjectBase') -> list['ObjectBase']:
        assert self.environment is not None, "Environment is not set, please call setup_environment first"
        ret = [deepcopy(obj) for obj in self.environment if ObjectBase.is_right(view, self, obj)]

        return ret

    def left(self, view: 'ObjectBase') -> list['ObjectBase']:
        assert self.environment is not None, "Environment is not set, please call setup_environment first"
        ret = [deepcopy(obj) for obj in self.environment if ObjectBase.is_left(view, self, obj)]

        return ret

    def opposite(self, view: 'ObjectBase') -> list['ObjectBase']:
        assert self.environment is not None, "Environment is not set, please call setup_environment first"
        ret = [deepcopy(obj) for obj in self.environment if ObjectBase.is_opposite(view, self, obj)]

        return ret

    def back(self, view: 'ObjectBase') -> list['ObjectBase']:
        assert self.environment is not None, "Environment is not set, please call setup_environment first"
        ret = [deepcopy(obj) for obj in self.environment if ObjectBase.is_back(view, self, obj)]

        return ret
    
    def behind(self, view: 'ObjectBase') -> list['ObjectBase']:
        assert self.environment is not None, "Environment is not set, please call setup_environment first"
        ret = [deepcopy(obj) for obj in self.environment if ObjectBase.is_behind(view, self, obj)]

        return ret

    @staticmethod
    def is_right(view: 'ObjectBase', source: 'ObjectBase', target: 'ObjectBase') -> bool:
        if view.direction() == 'up':
            return target.position()[0] > source.position()[0]
        elif view.direction() == 'down':
            return target.position()[0] < source.position()[0]
        elif view.direction() == 'left':
            return target.position()[1] < source.position()[1]
        elif view.direction() == 'right':
            return target.position()[1] > source.position()[1]
        else:
            raise ValueError(f"Invalid direction: {view.direction()}")

    @staticmethod
    def is_left(view: 'ObjectBase', source: 'ObjectBase', target: 'ObjectBase') -> bool:
        if view.direction() == 'up':
            return target.position()[0] < source.position()[0]
        elif view.direction() == 'down':
            return target.position()[0] > source.position()[0]
        elif view.direction() == 'left':
            return target.position()[1] > source.position()[1]
        elif view.direction() == 'right':
            return target.position()[1] < source.position()[1]
        else:
            raise ValueError(f"Invalid direction: {view.direction()}")

    @staticmethod
    def is_front(view: 'ObjectBase', source: 'ObjectBase', target: 'ObjectBase') -> bool:
        if view.direction() == 'up' or view.direction() == 'down':
            if view.position()[1] <= source.position()[1]:
                return target.position()[1] < source.position()[1]
            else:
                return target.position()[1] > source.position()[1]
        elif view.direction() == 'left' or view.direction() == 'right':
            if view.position()[0] <= source.position()[0]:
                return target.position()[0] < source.position()[0]
            else:
                return target.position()[0] > source.position()[0]
        else:
            raise ValueError(f"Invalid direction: {view.direction()}")

    @staticmethod
    def is_opposite(view: 'ObjectBase', source: 'ObjectBase', target: 'ObjectBase') -> bool:
        return ObjectBase.is_front(view, source, target)

    @staticmethod
    def is_back(view: 'ObjectBase', source: 'ObjectBase', target: 'ObjectBase') -> bool:
        if view.direction() == 'up' or view.direction() == 'down':
            if view.position()[1] <= source.position()[1]:
                return target.position()[1] > source.position()[1]
            else:
                return target.position()[1] < source.position()[1]
        elif view.direction() == 'left' or view.direction() == 'right':
            if view.position()[0] <= source.position()[0]:
                return target.position()[0] > source.position()[0]
            else:
                return target.position()[0] < source.position()[0]
        else:
            raise ValueError(f"Invalid direction: {view.direction()}")

    @staticmethod
    def is_behind(view: 'ObjectBase', source: 'ObjectBase', target: 'ObjectBase') -> bool:
        return ObjectBase.is_back(view, source, target)

class ObjectContext(ObjectBase):
    def __init__(self, name: str, position: tuple[int, int], direction: str):
        super().__init__(name, position, direction)

class ViewContext(ObjectBase):
    def __init__(self, name: str, position: tuple[int, int], direction: str):
        super().__init__(name, position, direction)

class AgentContext(ObjectBase):
    def __init__(self, name: str, position: tuple[int, int], direction: str):
        self.original_answer = None
        self.extracted_answer = None
        super().__init__(name, position, direction)

    def goto(self, position: tuple[int, int]) -> None:
        self._position = position

    def face(self, direction: str) -> None:
        self._direction = direction

    def answer(self, answer: str) -> None:
        self.original_answer = answer

def turn_right(face: str) -> str:
    """Turn the agent to the right direction."""
    if face == 'up':
        return 'right'
    elif face == 'right':
        return 'down'
    elif face == 'down':
        return 'left'
    elif face == 'left':
        return 'up'
    else:
        raise ValueError(f"Invalid face: {face}")

def turn_left(face: str) -> str:
    """Turn the agent to the left direction."""
    if face == 'up':
        return 'left'
    elif face == 'left':
        return 'down'
    elif face == 'down':
        return 'right'
    elif face == 'right':
        return 'up'
    else:
        raise ValueError(f"Invalid face: {face}")

def turn_back(face: str) -> str:
    """Turn the agent to the back direction."""
    if face == 'up':
        return 'down'
    elif face == 'down':
        return 'up'
    elif face == 'left':
        return 'right'
    elif face == 'right':
        return 'left'
    else:
        raise ValueError(f"Invalid face: {face}")

def go_straight(face: str) -> tuple[int, int]:
    """Go the agent to the straight direction."""
    if face == 'up':
        return (0, -1)
    elif face == 'down':
        return (0, 1)
    elif face == 'left':
        return (-1, 0)
    elif face == 'right':
        return (1, 0)
    else:
        raise ValueError(f"Invalid face: {face}")

def move_forward(face: str) -> tuple[int, int]:
    """Step the agent forward in the current direction."""
    return go_straight(face)
