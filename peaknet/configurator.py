from contextlib import contextmanager


class Configurator(dict):
    """
    Poorman's configurator, basically like AttrDict.

    Reference: https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/collections.py
    """
    __auto_create = False

    def __getattr__(self, attr):
        if attr not in self:
            if self.__class__.__auto_create:
                self[attr] = Configurator()
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{attr}'. "
                    "If you're trying to auto-create attributes, consider using the 'enable_auto_create' context manager in a with-statement."
                )
        return self[attr]


    def __setattr__(self, attr, value):
        if attr == "_Configurator__auto_create":  # Use name mangling for private attributes
            raise AttributeError("'_auto_create' is read-only!")
        self[attr] = value    # Using __setitem__ under the hood


    @contextmanager
    def enable_auto_create(self):
        original_state = self.__class__.__auto_create
        self.__class__.__auto_create = True
        try:
            yield self
        finally:
            self.__class__.__auto_create = original_state


    def to_dict(self):
        result = {}
        for key, value in self.items():
            if isinstance(value, Configurator):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


    @classmethod
    def from_dict(cls, data):
        instance = cls()
        for key, value in data.items():
            if isinstance(value, dict):
                instance[key] = cls.from_dict(value)
            else:
                instance[key] = value

        return instance
