# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A2A protocol support for ADK.

This package provides support for the Agent-to-Agent (A2A) protocol,
including missing proto types and converters.
"""

from __future__ import annotations

# Import types that are missing from a2a-sdk
from .types import ListTasksRequest
from .types import ListTasksResponse

# Re-export commonly used types from a2a-sdk for convenience
# These types exist in a2a-sdk but may be under different names
try:
  from a2a.types import GetAuthenticatedExtendedCardRequest
  from a2a.types import TaskResubscriptionRequest

  # Provide aliases matching the proto spec names
  GetExtendedAgentCardRequest = GetAuthenticatedExtendedCardRequest
  SubscribeToTaskRequest = TaskResubscriptionRequest
except ImportError:
  # If a2a-sdk doesn't have these types, they won't be available
  pass

__all__ = [
    'ListTasksRequest',
    'ListTasksResponse',
    'GetExtendedAgentCardRequest',
    'GetAuthenticatedExtendedCardRequest',
    'SubscribeToTaskRequest',
    'TaskResubscriptionRequest',
]
