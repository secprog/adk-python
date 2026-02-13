# Copyright 2025 Google LLC
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

"""A2A protocol types for ADK.

This module provides Pydantic models for A2A protocol message types that are
missing from the a2a-sdk package, along with convenient type aliases for types
that exist under different names in the SDK.

The types defined here correspond to messages in the official A2A proto
specification: https://github.com/a2aproject/A2A/blob/main/specification/grpc/a2a.proto
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class ListTasksRequest(BaseModel):
  """Parameters for listing tasks with optional filtering criteria.

  Corresponds to ListTasksRequest in the A2A proto specification.
  """

  tenant: Optional[str] = Field(
      default=None,
      description='Optional tenant, provided as a path parameter.',
  )
  context_id: Optional[str] = Field(
      default=None,
      description=(
          'Filter tasks by context ID to get tasks from a specific conversation'
          ' or session.'
      ),
  )
  status: Optional[str] = Field(
      default=None,
      description='Filter tasks by their current status state.',
  )
  page_size: Optional[int] = Field(
      default=None,
      description=(
          'Maximum number of tasks to return. Must be between 1 and 100.'
          ' Defaults to 50 if not specified.'
      ),
  )
  page_token: Optional[str] = Field(
      default=None,
      description=(
          'Token for pagination. Use the next_page_token from a previous'
          ' ListTasksResponse.'
      ),
  )
  history_length: Optional[int] = Field(
      default=None,
      description=(
          "The maximum number of messages to include in each task's history."
      ),
  )
  last_updated_after: Optional[int] = Field(
      default=None,
      description=(
          'Filter tasks updated after this timestamp (milliseconds since'
          ' epoch).'
      ),
  )
  include_artifacts: Optional[bool] = Field(
      default=None,
      description=(
          'Whether to include artifacts in the returned tasks. Defaults to'
          ' false to reduce payload size.'
      ),
  )


class ListTasksResponse(BaseModel):
  """Result object for tasks/list method.

  Contains an array of tasks and pagination information.
  Corresponds to ListTasksResponse in the A2A proto specification.
  """

  tasks: list = Field(
      default_factory=list,
      description='Array of tasks matching the specified criteria.',
  )
  next_page_token: str = Field(
      default='',
      description=(
          'Token for retrieving the next page. Empty string if no more results.'
      ),
  )
  page_size: int = Field(
      description='The size of page requested.',
  )
  total_size: int = Field(
      description='Total number of tasks available (before pagination).',
  )
