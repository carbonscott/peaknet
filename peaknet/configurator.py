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


    def get_value(self, key):
        """
        Retrieve a nested key using dot-separated notation.
        For example, for a key like "a.b.c", it fetches self['a']['b']['c'] if it exists, otherwise returns default.
        """
        keys  = key.split('.')
        value = self
        for k in keys:
            if k in value:
                value = value[k]
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{k}'. "
                )

        return value


    def set_value(self, key, value):
        """
        Set a nested key using dot-separated notation.
        For example, for a key like "a.b.c", it sets value to self['a']['b']['c'].
        Creates necessary nested Configurators if they don't exist.
        """
        keys = key.split('.')
        current_value = self
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                current_value[k] = value
            else:
                if k not in current_value or not isinstance(current_value[k], Configurator):
                    if self.__class__.__auto_create:
                        current_value[k] = Configurator()
                    else:
                        raise AttributeError(
                            f"'{type(self).__name__}' object has no attribute '{k}'. "
                            "If you're trying to auto-create attributes, consider using the 'enable_auto_create' context manager in a with-statement."
                        )
                current_value = current_value[k]
