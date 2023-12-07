from contextlib import contextmanager

class Configurator(dict):
    """
    Poorman's configurator, basically like AttrDict.

    Reference: https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/collections.py
    """
    __auto_create = False    # it is `_Configurator__auto_create` under the hood (name mangling)

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
        # Try to change __auto_create outside the class???
        if attr == "_Configurator__auto_create":
            raise AttributeError("'__auto_create' is read-only!")

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
        """
        Hierarchically change all nodes to a dictionary.
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, Configurator):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


    @classmethod
    def from_dict(cls, data):
        """
        Recursively create all nodes from a dictionary.
        """
        instance = cls()
        for key, value in data.items():
            if isinstance(value, dict):
                instance[key] = cls.from_dict(value)
            else:
                instance[key] = value

        return instance
