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

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: spaces.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder


# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import cogment_lab.generated.ndarray_pb2 as ndarray__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0cspaces.proto\x12\x12\x63ogment_lab.spaces\x1a\rndarray.proto"$\n\x08\x44iscrete\x12\t\n\x01n\x18\x01 \x01(\x05\x12\r\n\x05start\x18\x02 \x01(\x05"Z\n\x03\x42ox\x12(\n\x03low\x18\x02 \x01(\x0b\x32\x1b.cogment_lab.nd_array.Array\x12)\n\x04high\x18\x03 \x01(\x0b\x32\x1b.cogment_lab.nd_array.Array"5\n\x0bMultiBinary\x12&\n\x01n\x18\x01 \x01(\x0b\x32\x1b.cogment_lab.nd_array.Array":\n\rMultiDiscrete\x12)\n\x04nvec\x18\x01 \x01(\x0b\x32\x1b.cogment_lab.nd_array.Array"|\n\x04\x44ict\x12\x31\n\x06spaces\x18\x01 \x03(\x0b\x32!.cogment_lab.spaces.Dict.SubSpace\x1a\x41\n\x08SubSpace\x12\x0b\n\x03key\x18\x01 \x01(\t\x12(\n\x05space\x18\x02 \x01(\x0b\x32\x19.cogment_lab.spaces.Space"\x89\x02\n\x05Space\x12\x30\n\x08\x64iscrete\x18\x01 \x01(\x0b\x32\x1c.cogment_lab.spaces.DiscreteH\x00\x12&\n\x03\x62ox\x18\x02 \x01(\x0b\x32\x17.cogment_lab.spaces.BoxH\x00\x12(\n\x04\x64ict\x18\x03 \x01(\x0b\x32\x18.cogment_lab.spaces.DictH\x00\x12\x37\n\x0cmulti_binary\x18\x04 \x01(\x0b\x32\x1f.cogment_lab.spaces.MultiBinaryH\x00\x12;\n\x0emulti_discrete\x18\x05 \x01(\x0b\x32!.cogment_lab.spaces.MultiDiscreteH\x00\x42\x06\n\x04kindb\x06proto3'
)

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "spaces_pb2", globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _DISCRETE._serialized_start = 51
    _DISCRETE._serialized_end = 87
    _BOX._serialized_start = 89
    _BOX._serialized_end = 179
    _MULTIBINARY._serialized_start = 181
    _MULTIBINARY._serialized_end = 234
    _MULTIDISCRETE._serialized_start = 236
    _MULTIDISCRETE._serialized_end = 294
    _DICT._serialized_start = 296
    _DICT._serialized_end = 420
    _DICT_SUBSPACE._serialized_start = 355
    _DICT_SUBSPACE._serialized_end = 420
    _SPACE._serialized_start = 423
    _SPACE._serialized_end = 688
# @@protoc_insertion_point(module_scope)
