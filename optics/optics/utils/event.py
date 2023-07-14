from __future__ import annotations

from typing import Callable, List, Generator, Union, Sequence
from multiprocessing import RLock
import queue

__all__ = ['Event', 'event', 'handler', 'dont_wait_for', 'wait_for']


class Event:
    Handler = Callable

    def __init__(self, name: str = ''):
        self._lock = RLock()
        self.__name = name
        self.__handlers: List[Event.Handler] = []

    @property
    def name(self) -> str:
        return self.__name

    def __iter__(self) -> Generator[Handler]:
        return (_ for _ in self.__handlers)

    def __call__(self, *args, **kwargs):
        """
        Report event to all subscribers
        """
        with self._lock:
            for handler in self.__handlers:
                handler(*args, **kwargs)

    def __iadd__(self, handlers: Union[Handler, Sequence[Handler]]) -> Event:
        """
        Subscribe to an event.
        :param handlers: The functions to subsribe.
        """
        with self._lock:
            if not isinstance(handlers, Sequence):
                handlers = [handlers]
            self.__handlers += handlers
            return self

    def __isub__(self, handlers: Union[Handler, Sequence[Handler]]) -> Event:
        """
        Unsubscribe from an event.
        :param handlers: The functions to unsubscribe.
        """
        with self._lock:
            if not isinstance(handlers, Sequence):
                handlers = [handlers]
            for _ in handlers:
                self.__handlers.remove(_)
            return self

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'


def event(getter_method: Callable[..., Event]):
    def wrapper(self):
        return getter_method(self)

    def wrapper_setter(self, new_event: Event):
        event = getter_method(self)
        if new_event != event:
            raise AttributeError(f'Cannot set the {event} property to a new {new_event}!')

    return property(wrapper, wrapper_setter, None, 'Test')


def handler(event: Event):
    def wrapper(func: Callable):
        event.__iadd__(func)

    return wrapper


def dont_wait_for(event: Event):
    def wrapper(func: Callable):
        def inner(*args, **kwargs):
            def one_off_func(*args_ignored, **kwargs_ignored):
                func(*args, **kwargs)
                event.__isub__(one_off_func)
            event.__iadd__(one_off_func)
        return inner
    return wrapper


def wait_for(event: Event):
    def wrapper(func: Callable):
        def inner(*args, **kwargs):
            result = queue.Queue(maxsize=1)
            def one_off_func(*args_ignored, **kwargs_ignored):
                result.put(func(*args, **kwargs))
                event.__isub__(one_off_func)

            event.__iadd__(one_off_func)
            return result.get()
        return inner
    return wrapper


if __name__ == '__main__':
    from threading import Thread
    import time

    class Win:
        """Just a test class"""
        def __init__(self):
            self.__on_move = Event('Move')
            self.__on_delete = Event('Delete')
            self.__deleted = False
            self.__pos = None

        @event
        def on_move(self) -> Event:
            return self.__on_move

        @event
        def on_delete(self) -> Event:
            return self.__on_delete

        def move(self, new_pos: Sequence):
            if new_pos != self.__pos:
                self.__pos = new_pos
                self.on_move(new_pos)

        def __del__(self):
            if not self.__deleted:
                self.__deleted = True
                self.on_delete()


    win1 = Win()
    win2 = Win()

    win1.on_delete += lambda: print('deleted win 1')
    win2.on_delete += lambda: print('deleted win 2')
    win1.on_move += lambda p: print(f'moved win 1 to {p}')
    win2.on_move += lambda p: print(f'moved win 2 to {p}')
    win1.on_move += win2.move
    win2.on_move += win1.move
    win1.on_delete += lambda: win2.__del__()
    win2.on_delete += lambda: win1.__del__()

    @wait_for(win1.on_move)
    def delegated_function(x):
        return x*5

    @dont_wait_for(win1.on_move)
    def delegated_procedure(x):
        print(f'Side effect {x*5}!')

    thread = Thread(target=lambda: print(delegated_function('a')))

    # for _ in win2.on_move:
    #     print(_)
    win2.on_move -= win1.move
    # win1.on_move = Event('Break!')

    win2.move((3, 4))
    win1.move((1, 2))
    delegated_procedure('B')
    win1.move((2, 3))
    for idx in range(5):
        if idx == 2:
            thread.start()
        time.sleep(1)
        win1.move((idx, idx))

    win2.__del__()
