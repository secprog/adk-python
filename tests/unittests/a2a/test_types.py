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

"""Tests for A2A protocol types."""

from __future__ import annotations

from google.adk.a2a.types import ListTasksRequest
from google.adk.a2a.types import ListTasksResponse
import pytest


class TestListTasksRequest:
  """Tests for ListTasksRequest model."""

  def test_defaults(self):
    """Test ListTasksRequest with all defaults."""
    request = ListTasksRequest()
    assert request.tenant is None
    assert request.context_id is None
    assert request.status is None
    assert request.page_size is None
    assert request.page_token is None
    assert request.history_length is None
    assert request.last_updated_after is None
    assert request.include_artifacts is None

  def test_with_all_fields(self):
    """Test ListTasksRequest with all fields populated."""
    request = ListTasksRequest(
        tenant='my-tenant',
        context_id='ctx-123',
        status='completed',
        page_size=25,
        page_token='token-abc',
        history_length=5,
        last_updated_after=1234567890,
        include_artifacts=True,
    )
    assert request.tenant == 'my-tenant'
    assert request.context_id == 'ctx-123'
    assert request.status == 'completed'
    assert request.page_size == 25
    assert request.page_token == 'token-abc'
    assert request.history_length == 5
    assert request.last_updated_after == 1234567890
    assert request.include_artifacts is True

  def test_with_partial_fields(self):
    """Test ListTasksRequest with only some fields."""
    request = ListTasksRequest(
        context_id='ctx-456',
        page_size=10,
    )
    assert request.context_id == 'ctx-456'
    assert request.page_size == 10
    assert request.tenant is None
    assert request.status is None

  def test_serialization(self):
    """Test that ListTasksRequest can be serialized."""
    request = ListTasksRequest(
        context_id='ctx-789',
        page_size=50,
        include_artifacts=False,
    )
    # Should be able to convert to dict
    data = request.model_dump()
    assert data['context_id'] == 'ctx-789'
    assert data['page_size'] == 50
    assert data['include_artifacts'] is False

  def test_json_serialization(self):
    """Test that ListTasksRequest can be serialized to JSON."""
    request = ListTasksRequest(
        tenant='test-tenant',
        context_id='ctx-abc',
        page_size=20,
    )
    json_str = request.model_dump_json()
    assert 'test-tenant' in json_str
    assert 'ctx-abc' in json_str


class TestListTasksResponse:
  """Tests for ListTasksResponse model."""

  def test_required_fields(self):
    """Test ListTasksResponse with required fields only."""
    response = ListTasksResponse(
        tasks=[],
        next_page_token='',
        page_size=50,
        total_size=0,
    )
    assert response.tasks == []
    assert response.next_page_token == ''
    assert response.page_size == 50
    assert response.total_size == 0

  def test_with_tasks(self):
    """Test ListTasksResponse with task data."""
    task1 = {'id': 'task-1', 'status': 'completed'}
    task2 = {'id': 'task-2', 'status': 'working'}
    response = ListTasksResponse(
        tasks=[task1, task2],
        next_page_token='next-token-xyz',
        page_size=2,
        total_size=10,
    )
    assert len(response.tasks) == 2
    assert response.tasks[0] == task1
    assert response.tasks[1] == task2
    assert response.next_page_token == 'next-token-xyz'
    assert response.page_size == 2
    assert response.total_size == 10

  def test_empty_tasks_with_pagination(self):
    """Test ListTasksResponse with no tasks but pagination info."""
    response = ListTasksResponse(
        tasks=[],
        next_page_token='',
        page_size=25,
        total_size=0,
    )
    assert response.tasks == []
    assert response.next_page_token == ''
    assert response.page_size == 25
    assert response.total_size == 0

  def test_missing_required_field(self):
    """Test that ListTasksResponse requires page_size field."""
    with pytest.raises(Exception):  # Pydantic ValidationError
      ListTasksResponse(
          tasks=[],
          next_page_token='',
          total_size=0,
          # Missing page_size
      )

  def test_serialization(self):
    """Test that ListTasksResponse can be serialized."""
    response = ListTasksResponse(
        tasks=[{'id': 'task-1'}],
        next_page_token='token',
        page_size=1,
        total_size=5,
    )
    data = response.model_dump()
    assert len(data['tasks']) == 1
    assert data['next_page_token'] == 'token'
    assert data['page_size'] == 1
    assert data['total_size'] == 5

  def test_json_serialization(self):
    """Test that ListTasksResponse can be serialized to JSON."""
    response = ListTasksResponse(
        tasks=[],
        next_page_token='',
        page_size=100,
        total_size=250,
    )
    json_str = response.model_dump_json()
    assert '100' in json_str
    assert '250' in json_str

  def test_default_factory_for_tasks(self):
    """Test that tasks field uses default_factory for empty list."""
    response = ListTasksResponse(
        next_page_token='',
        page_size=10,
        total_size=0,
    )
    # Should default to empty list
    assert response.tasks == []
    assert isinstance(response.tasks, list)


class TestA2ATypeAliases:
  """Tests for type aliases and re-exports from a2a-sdk."""

  def test_type_aliases_exist(self):
    """Test that type aliases are properly defined."""
    # These should be importable
    # They should be aliases to the actual a2a-sdk types
    from google.adk.a2a import GetAuthenticatedExtendedCardRequest
    from google.adk.a2a import GetExtendedAgentCardRequest
    from google.adk.a2a import SubscribeToTaskRequest
    from google.adk.a2a import TaskResubscriptionRequest

    assert GetExtendedAgentCardRequest is GetAuthenticatedExtendedCardRequest
    assert SubscribeToTaskRequest is TaskResubscriptionRequest

  def test_list_tasks_types_exist(self):
    """Test that ListTasks types are properly exported."""
    from google.adk.a2a import ListTasksRequest as Req
    from google.adk.a2a import ListTasksResponse as Resp

    # Should be able to instantiate
    req = Req()
    resp = Resp(page_size=10, total_size=0)
    assert req is not None
    assert resp is not None
