from contextlib import contextmanager
from dataclasses import make_dataclass, is_dataclass, fields


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


    @staticmethod
    def define_dataclass_from_dict(input_dict, class_name = 'Config'):
        """
        Dynamically create a dataclass from a dictionary, with nested
        dictionary nodes converted into nested dataclass nodes.

        :param dict_data: The dictionary representing the structure of dataclass.
        :param class_name: The name for the generated dataclass.
        :return: A new dataclass type corresponding to the structure of dict_data.

        Main difference between dataclass nodes and dictionary nodes:
        both nested dataclass and nested dictionary can be represented by a
        graph with nodes and edges.  The difference is that each new dataclass
        node has a type while each new dictioanry node is stil a dictionary.

        For example, suppose we have a dataclass node L0_A (meaning level 0
        type A) that has one child node who's also a dataclass node with the
        name L1_A.

        dataclass node L0_A:      |    dict node:
        - dataclass node L1_A:    |    - dict node:
            - field 1             |        - field 1
            - field 2             |        - field 2
        - field                   |    - field
        """
        new_fields = []
        for key, value in input_dict.items():
            # Need a new dataclass node???
            if isinstance(value, dict):
                field_type = Configurator.define_dataclass_from_dict(value, class_name = f'{key}Config')

            # Otherwise...
            else:
                field_type = type(value)
            new_fields.append((key, field_type))

        new_dataclass = make_dataclass(class_name, new_fields)

        return new_dataclass


    @staticmethod
    def instantiate_dataclass_from_dict(custom_dataclass, input_dict):
        """
        Recursively convert a nested dictionary into an instance of a dataclass.

        :param custom_dataclass: The dataclass type to convert to.
        :param input_dict: The dictionary to convert from.
        :return: An instance of custom_dataclass if it's a dataclass, otherwise the original data.
        """
        # Check if custom_dataclass is a dataclass
        if is_dataclass(custom_dataclass):
            # Create an empty dictionary to store the new field values...
            new_field_values = {}

            # Iterate over each field defined in the dataclass...
            for field in fields(custom_dataclass):
                # Extract field's name, type and value...
                field_name  = field.name
                field_type  = field.type
                field_value = input_dict.get(field_name)

                # Keep instantiating a new dataclass node with the value of the dictionary node???
                if is_dataclass(field_type):
                    new_field_value = Configurator.instantiate_dataclass_from_dict(field_type, field_value)

                # Otherwise, save the field value as is...
                else:
                    new_field_value = field_value

                # Store the converted field value...
                new_field_values[field_name] = new_field_value

            # Create a new instance of the dataclass with the converted fields
            return custom_dataclass(**new_field_values)
        else:
            # If custom_dataclass is not a dataclass, just return the data as is
            return input_dict


    def to_dataclass(self):
        internal_dict = self.to_dict()

        PeakNetConfig         = Configurator.define_dataclass_from_dict(internal_dict, 'PeakNetConfig')
        peaknet_static_config = Configurator.instantiate_dataclass_from_dict(PeakNetConfig, internal_dict)

        return peaknet_static_config
