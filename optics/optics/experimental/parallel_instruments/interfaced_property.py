from typing import List, Dict, Optional, Type, Callable, Sequence, Union, Set, Generic, TypeVar
import inspect
from abc import ABC

from optics.experimental.parallel_instruments import log

T = TypeVar('T')


def prop(f):
    def wrapped_f(obj, value, /):
        if value < 0:
            raise ValueError('Only positive values can be used')
        return f(obj, value)
    return wrapped_f


class Prop:
    def __init__(self, value: int, mutable: bool = False):
        self.__private_name = None
        self.value = value
        self.__mutable = mutable

    def __set_name__(self, owner, name):
        self.__private_name = f'__{name}'

    def __get__(self, obj, cls):
        return getattr(obj, self.__private_name)

    def __set__(self, obj, value):
        if self.__mutable:
            setattr(obj, self.__private_name, value)
        else:
            raise NotImplementedError(f'Cannot change property value.')

    @property
    def mutable(self) -> bool:
        return self.__mutable

    @mutable.setter
    def mutable(self, value: bool):
        self.__mutable = value


class Person:
    age = Prop(0)

    @prop
    def aged(self, number: int) -> int:
        self.age = number + self.age
        return self.age


a = Person()
b = Person()
a.age = 3
b.age = 5
a.age = 4
print(f'a.age = {a.age}, b.age = {b.age}')

a.aged(2)
b.aged(2)
print(f'a.age = {a.age}, b.age = {b.age}')
try:
    a.aged(-1)
except ValueError as ve:
    print(f'Indeed, {ve}')
print(f'a.age = {a.age}, b.age = {b.age}')
try:
    b.age = -1
except ValueError as ve:
    print(f'Indeed, {ve}')
print(f'a.age = {a.age}, b.age = {b.age}')

exit()


class ValueProp(property, Generic[T]):
    """A class to represent values that carry extra information and can act upon changes."""
    def __init__(self, value: T, settable: bool = True, accepted: Union[slice, range, Set, Sequence, None] = None):
        self.__value: T = value
        self.__settable: bool = settable
        self.settable = settable
        self.__accepted = accepted

        self.__owner = None
        self.__name = None

        def fget(self) -> T:
            log.warning(f'Getting {self}')
            return self.__value

        def fset(self, value: T):
            log.warning(f'Setting {self} to {value}')
            self.__value = value

        super().__init__(fget=fget, fset=fset)

    def __setter(self) -> Callable:
        def wrap_set(f: Callable) -> Callable:
            def wrapped_set(s, v):
                return f(s, v)
            return wrapped_set
        wrap_set.accepted = self.accepted[0]
        return wrap_set

    def __set_name__(self, owner, name):
        self.__owner = owner
        self.__name = name
        owner.__dict__[name] = self.__setter

    @property
    def settable(self) -> bool:
        return self.__settable

    @settable.setter
    def settable(self, value: bool):
        self.__settable = value
        if value:
            self.__set__ = super().__set__
        else:
            self.__set__ = None

    @property
    def accepted(self) -> Union[slice, range, Set, Sequence, None]:
        return self.__accepted


def shared(*args, **kwargs):
    accepted = dict(enumerate(args)) | kwargs

    def wrap_method(f: Callable) -> Callable:
        """The function that produces wrapped method."""
        def wrapped_method(self, *margs, **mkwargs):
            """The new method that gets called."""
            return f(self, *margs, **mkwargs)
        wrapped_method.accepted = accepted

        def setter(self) -> Callable:
            def wrap_set(f: Callable) -> Callable:
                def wrapped_set(s, v):
                    return f(s, v)
                return wrapped_set
            wrap_set.accepted = accepted[0]
            return wrap_set

        wrapped_method.setter = setter

        return wrapped_method

    return wrap_method


class Component:
    id: int = 5

    def __init__(self, name: str = 'Me'):
        self.__name = name
        self.__number = 0.0

    @shared(slice(0, 10, 1))
    def number(self) -> float:
        return self.__number

    @number.setter
    def number(self, value: float):
        self.__number = value

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str):
        self.__name = value

    @shared(slice(4), range(3), id=slice(0, 10, 1), greeting=['Hi', 'Hello', 'Hola'])
    def say(self, greeting: str = 'Hello', id: Optional[int] = None) -> str:
        if id is None:
            id = ''
        else:
            id = f'-{id}'
        msg = f'{greeting}{id}, my name is {self.name}!'
        log.info('"' + msg + '"')
        return msg

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'


class ABCInterfacedComponent(ABC):
    _component = None
    _exposed_attributes = dict()
    _exposed_attributes_type = dict()

    def __init__(self, component):
        self._component = component


class ExposedComponent(ABCInterfacedComponent):
    def __init__(self, component):
        super(ExposedComponent, self).__init__(component)
        for k, v in inspect.getmembers(component):
            if not inspect.isbuiltin(v):
                if not k.startswith('_'):
                    if callable(v):
                        # log.info(f'Method {k}(): {repr(v)}, type = {type(v)}')
                        # s = inspect.signature(v)
                        # self._exposed_attributes_type[k] = s.return_annotation
                        # for p, pv in s.parameters.items():
                        #     log.info(f'           {pv.name}: {pv.annotation} = {pv.default}')
                        self._exposed_attributes[k] = v
                    else:
                        log.info(f'Property {k}: {repr(v)}, type = {type(v)}')
                        attribute = inspect.getattr_static(component, k)
                        if isinstance(attribute, property):
                            self._exposed_attributes[k] = attribute
                            settable = attribute.fset is not None
                            prop_type = inspect.signature(attribute.fget).return_annotation
                            # self._exposed_attributes_type[k] = s.return_annotation
                            log.info(f'    {k} has type {prop_type}, settable = {settable}.')
                        else:
                            # log.info(f'{attribute} is NOT a property!')
                            wrapped_attribute = property(
                                                    fset=lambda: getattr(self._component, k),
                                                    fget=lambda _: setattr(self._component, k, _))

                            signature = inspect.signature(wrapped_attribute.fset)
                            self._exposed_attributes[k] = wrapped_attribute
                        # log.info(f'Property {k}: {repr(prop)},  => {dir(prop)}')
            #     else:
            #         log.warning(f'Skipping {k}: {v}')
                elif k == '__init__':
                    self._exposed_attributes[k] = v

    @property
    def __dict__(self) -> Dict[str, object]:
        return super(ExposedComponent, self)._exposed_attributes

    def __dir__(self) -> List[str]:
        return list(self.__dict__.keys())

    def __getattr__(self, item):
        obj = super(ExposedComponent, self) if item.startswith('_') or item not in super(ExposedComponent, self)._exposed_attributes else self._component
        return obj.__getattribute__(item)

    def __setattr__(self, item, value):
        obj = super(ExposedComponent, self) if item.startswith('_') or item not in super(ExposedComponent, self)._exposed_attributes else self._component
        return obj.__setattr__(item, value)

    def __delattr__(self, item):
        if item.startswith('_') or item not in super()._exposed_attributes:
            return super(ExposedComponent, self).__delattr__(item)
        else:
            raise AttributeError(f'Cannot delete attribute {item}.')

    def __str__(self) -> str:
        return f'Exposed-{self._component}'


class Interface(ExposedComponent):
    def desc_constructor(self) -> str:
        constructor = self._exposed_attributes['__init__']
        signature = inspect.signature(constructor)
        arguments = [f'{pv.name}: {pv.annotation.__name__}' + (f' = {repr(pv.default)}' if pv.default != inspect._empty else '') for p, pv in signature.parameters.items()]
        arguments = ', '.join(arguments)
        return f'{self._component.__class__.__name__}({arguments})'

    def desc_properties(self) -> str:
        attr_descs = []
        for name, p in self._exposed_attributes.items():
            if isinstance(p, property):
                writable = p.fset is not None
                signature = inspect.signature(p.fget)
                type_desc = f': {signature.return_annotation.__name__}' if signature.return_annotation != inspect._empty else ''
                attr_descs.append(f'{name}{type_desc} = {getattr(self, name)}' + (' (writable)' if writable else ''))
        return '\n'.join(attr_descs)

    def desc_methods(self) -> str:
        attr_descs = []
        for name, m in self._exposed_attributes.items():
            if name != '__init__' and not isinstance(m, property):
                signature = inspect.signature(m)
                arguments = [f'{pv.name}: {pv.annotation.__name__}' + (f' = {repr(pv.default)}' if pv.default != inspect._empty else '') for p, pv in signature.parameters.items()]
                arguments = ', '.join(arguments)
                ret = f' -> {signature.return_annotation.__name__}' if signature.return_annotation != inspect._empty else ''
                attr_descs.append(f'{name}({arguments})' + ret)
        return '\n'.join(attr_descs)

    def __str__(self) -> str:
        return f'Interfaced-{self._component}'


if __name__ == '__main__':
    component = Component('Me')
    component.id = 3
    log.info(f'component = {component}, dir = {dir(component)}')
    log.info(f'component = {component}, dict = {component.__dict__}')

    reference = Interface(component)

    log.info(f'reference.id = {reference.id}')
    log.info(f'reference = {reference}')

    reference.id = 2
    reference.name = 'Test'

    log.info(f'reference.id = {reference.id}')
    log.info(f'reference = {reference}, dir = {dir(reference)}')
    log.info(f'reference = {reference}, dict = {reference.__dict__}')

    log.info(f'Constructor:\n {reference.desc_constructor()}')
    log.info(f'Methods:\n {reference.desc_methods()}')
    log.info(f'Properties:\n {reference.desc_properties()}')

    log.info(f'  reference.number = {reference.number}: {type(reference.number).__name__}')
    reference.number = 3.14
    log.info(f'->reference.number = {reference.number}: {type(reference.number).__name__}')

    log.info(f'  reference.say.accepted = {reference.say.accepted}')
    log.info(f'  reference.say() = {reference.say()}: {type(reference.say())}')

    log.info(f'  vars(reference)["number"].accepted = {vars(reference)["number"].accepted}')



