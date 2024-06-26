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

# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: data.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database


# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import cogment_lab.generated.ndarray_pb2 as ndarray__pb2
import cogment_lab.generated.spaces_pb2 as spaces__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ndata.proto\x12\x0b\x63ogment_lab\x1a\rndarray.proto\x1a\x0cspaces.proto\"\xd7\x01\n\x10\x45nvironmentSpecs\x12\x16\n\x0eimplementation\x18\x01 \x01(\t\x12\x12\n\nturn_based\x18\x02 \x01(\x08\x12\x13\n\x0bnum_players\x18\x03 \x01(\x05\x12\x34\n\x11observation_space\x18\x04 \x01(\x0b\x32\x19.cogment_lab.spaces.Space\x12/\n\x0c\x61\x63tion_space\x18\x05 \x01(\x0b\x32\x19.cogment_lab.spaces.Space\x12\x1b\n\x13web_components_file\x18\x06 \x01(\t\"s\n\nAgentSpecs\x12\x34\n\x11observation_space\x18\x01 \x01(\x0b\x32\x19.cogment_lab.spaces.Space\x12/\n\x0c\x61\x63tion_space\x18\x02 \x01(\x0b\x32\x19.cogment_lab.spaces.Space\"Y\n\x05Value\x12\x16\n\x0cstring_value\x18\x01 \x01(\tH\x00\x12\x13\n\tint_value\x18\x02 \x01(\x05H\x00\x12\x15\n\x0b\x66loat_value\x18\x03 \x01(\x02H\x00\x42\x0c\n\nvalue_type\"\xf1\x01\n\x11\x45nvironmentConfig\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0e\n\x06render\x18\x02 \x01(\x08\x12\x14\n\x0crender_width\x18\x03 \x01(\x05\x12\x0c\n\x04seed\x18\x04 \x01(\r\x12\x0f\n\x07\x66latten\x18\x05 \x01(\x08\x12\x41\n\nreset_args\x18\x06 \x03(\x0b\x32-.cogment_lab.EnvironmentConfig.ResetArgsEntry\x1a\x44\n\x0eResetArgsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12!\n\x05value\x18\x02 \x01(\x0b\x32\x12.cogment_lab.Value:\x02\x38\x01\"/\n\nHFHubModel\x12\x0f\n\x07repo_id\x18\x01 \x01(\t\x12\x10\n\x08\x66ilename\x18\x02 \x01(\t\"\xa4\x01\n\x0b\x41gentConfig\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12,\n\x0b\x61gent_specs\x18\x02 \x01(\x0b\x32\x17.cogment_lab.AgentSpecs\x12\x0c\n\x04seed\x18\x03 \x01(\r\x12\x10\n\x08model_id\x18\x04 \x01(\t\x12\x17\n\x0fmodel_iteration\x18\x05 \x01(\x05\x12\x1e\n\x16model_update_frequency\x18\x06 \x01(\x05\"\r\n\x0bTrialConfig\"\x88\x01\n\x0bObservation\x12*\n\x05value\x18\x01 \x01(\x0b\x32\x1b.cogment_lab.nd_array.Array\x12\x0e\n\x06\x61\x63tive\x18\x02 \x01(\x08\x12\r\n\x05\x61live\x18\x03 \x01(\x08\x12\x1b\n\x0erendered_frame\x18\x04 \x01(\x0cH\x00\x88\x01\x01\x42\x11\n\x0f_rendered_frame\":\n\x0cPlayerAction\x12*\n\x05value\x18\x01 \x01(\x0b\x32\x1b.cogment_lab.nd_array.Arrayb\x06proto3')



_ENVIRONMENTSPECS = DESCRIPTOR.message_types_by_name['EnvironmentSpecs']
_AGENTSPECS = DESCRIPTOR.message_types_by_name['AgentSpecs']
_VALUE = DESCRIPTOR.message_types_by_name['Value']
_ENVIRONMENTCONFIG = DESCRIPTOR.message_types_by_name['EnvironmentConfig']
_ENVIRONMENTCONFIG_RESETARGSENTRY = _ENVIRONMENTCONFIG.nested_types_by_name['ResetArgsEntry']
_HFHUBMODEL = DESCRIPTOR.message_types_by_name['HFHubModel']
_AGENTCONFIG = DESCRIPTOR.message_types_by_name['AgentConfig']
_TRIALCONFIG = DESCRIPTOR.message_types_by_name['TrialConfig']
_OBSERVATION = DESCRIPTOR.message_types_by_name['Observation']
_PLAYERACTION = DESCRIPTOR.message_types_by_name['PlayerAction']
EnvironmentSpecs = _reflection.GeneratedProtocolMessageType('EnvironmentSpecs', (_message.Message,), {
  'DESCRIPTOR' : _ENVIRONMENTSPECS,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:cogment_lab.EnvironmentSpecs)
  })
_sym_db.RegisterMessage(EnvironmentSpecs)

AgentSpecs = _reflection.GeneratedProtocolMessageType('AgentSpecs', (_message.Message,), {
  'DESCRIPTOR' : _AGENTSPECS,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:cogment_lab.AgentSpecs)
  })
_sym_db.RegisterMessage(AgentSpecs)

Value = _reflection.GeneratedProtocolMessageType('Value', (_message.Message,), {
  'DESCRIPTOR' : _VALUE,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:cogment_lab.Value)
  })
_sym_db.RegisterMessage(Value)

EnvironmentConfig = _reflection.GeneratedProtocolMessageType('EnvironmentConfig', (_message.Message,), {

  'ResetArgsEntry' : _reflection.GeneratedProtocolMessageType('ResetArgsEntry', (_message.Message,), {
    'DESCRIPTOR' : _ENVIRONMENTCONFIG_RESETARGSENTRY,
    '__module__' : 'data_pb2'
    # @@protoc_insertion_point(class_scope:cogment_lab.EnvironmentConfig.ResetArgsEntry)
    })
  ,
  'DESCRIPTOR' : _ENVIRONMENTCONFIG,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:cogment_lab.EnvironmentConfig)
  })
_sym_db.RegisterMessage(EnvironmentConfig)
_sym_db.RegisterMessage(EnvironmentConfig.ResetArgsEntry)

HFHubModel = _reflection.GeneratedProtocolMessageType('HFHubModel', (_message.Message,), {
  'DESCRIPTOR' : _HFHUBMODEL,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:cogment_lab.HFHubModel)
  })
_sym_db.RegisterMessage(HFHubModel)

AgentConfig = _reflection.GeneratedProtocolMessageType('AgentConfig', (_message.Message,), {
  'DESCRIPTOR' : _AGENTCONFIG,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:cogment_lab.AgentConfig)
  })
_sym_db.RegisterMessage(AgentConfig)

TrialConfig = _reflection.GeneratedProtocolMessageType('TrialConfig', (_message.Message,), {
  'DESCRIPTOR' : _TRIALCONFIG,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:cogment_lab.TrialConfig)
  })
_sym_db.RegisterMessage(TrialConfig)

Observation = _reflection.GeneratedProtocolMessageType('Observation', (_message.Message,), {
  'DESCRIPTOR' : _OBSERVATION,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:cogment_lab.Observation)
  })
_sym_db.RegisterMessage(Observation)

PlayerAction = _reflection.GeneratedProtocolMessageType('PlayerAction', (_message.Message,), {
  'DESCRIPTOR' : _PLAYERACTION,
  '__module__' : 'data_pb2'
  # @@protoc_insertion_point(class_scope:cogment_lab.PlayerAction)
  })
_sym_db.RegisterMessage(PlayerAction)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _ENVIRONMENTCONFIG_RESETARGSENTRY._options = None
  _ENVIRONMENTCONFIG_RESETARGSENTRY._serialized_options = b'8\001'
  _ENVIRONMENTSPECS._serialized_start=57
  _ENVIRONMENTSPECS._serialized_end=272
  _AGENTSPECS._serialized_start=274
  _AGENTSPECS._serialized_end=389
  _VALUE._serialized_start=391
  _VALUE._serialized_end=480
  _ENVIRONMENTCONFIG._serialized_start=483
  _ENVIRONMENTCONFIG._serialized_end=724
  _ENVIRONMENTCONFIG_RESETARGSENTRY._serialized_start=656
  _ENVIRONMENTCONFIG_RESETARGSENTRY._serialized_end=724
  _HFHUBMODEL._serialized_start=726
  _HFHUBMODEL._serialized_end=773
  _AGENTCONFIG._serialized_start=776
  _AGENTCONFIG._serialized_end=940
  _TRIALCONFIG._serialized_start=942
  _TRIALCONFIG._serialized_end=955
  _OBSERVATION._serialized_start=958
  _OBSERVATION._serialized_end=1094
  _PLAYERACTION._serialized_start=1096
  _PLAYERACTION._serialized_end=1154
# @@protoc_insertion_point(module_scope)
