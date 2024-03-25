# Copyright 2024 AI Redefined Inc. <dev+cogment@ai-r.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module


def import_object(class_name: str):
    """
    Imports an object from a module based on a string.
    If the argument is a string without any dots, it's taken from the current namespace.

    Args:
        class_name (str): The full path to the object e.g. "package.module.Class"
                          or the name of the object in the current namespace.

    Returns:
        object: The imported object
    """
    if "." not in class_name:
        return globals()[class_name]
    else:
        module_path, class_name = class_name.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
