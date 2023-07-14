import json
import numpy as np
import dataclasses
import logging

from optics.utils import Roi

log = logging.getLogger(__name__)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, range):
            return list(o)
        elif isinstance(o, np.ndarray):
            if o.dtype != complex:
                return o.tolist()
            else:
                return dict(real=o.real, imag=o.imag)
        else:
            return o.__dict__


def object_hook(dct: dict):
    if 'real' in dct and 'imag' in dct:
        return np.array(dct['real']) + 1j * np.array(dct['imag'])
    elif 'top_left' in dct and 'shape' in dct:
        return Roi(top_left=dct['top_left'], shape=dct['shape'])
    else:
        for key, value in dct.items():
            if isinstance(value, list):
                try:
                    v = np.array(value)
                    dct[key] = v
                except ValueError as e:
                    pass
        return dct


class JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        if 'object_hook' not in kwargs or kwargs['object_hook'] is None:
            kwargs['object_hook'] = object_hook
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    @dataclasses.dataclass
    class Settings:
        a: int = 0
        b: str = ''
        c: list = dataclasses.field(default_factory=list)
        d: range = range(0)
        e: np.ndarray = np.ndarray([])
        f: Roi = Roi()


    arr = np.random.randn(5) + 1j * np.random.randn(5)
    s = Settings(a=4, b='Test', c=[[1, 2], [3, 4]], d=range(4), e=arr, f=Roi([10, 20, 5, 5]))

    result = json.dumps(s, indent=2, sort_keys=True, cls=JSONEncoder)
    js_dict = json.loads(result, cls=JSONDecoder)
    s2 = Settings()
    json.JSONDecoder
    s2.__dict__ = js_dict

    log.info(s2)
