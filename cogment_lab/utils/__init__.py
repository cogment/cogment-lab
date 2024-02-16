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

from cogment_lab.utils.grpc import extend_actor_config
from cogment_lab.utils.import_class import import_object
from cogment_lab.utils.runners import process_cleanup, setup_logging
from cogment_lab.utils.trial_utils import (
    TrialData,
    format_data_multiagent,
    get_actor_params,
)
