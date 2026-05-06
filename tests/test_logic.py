import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from intent_parser import IntentParser
from semantic_mapper import SemanticMapper
from models import NodeData

@pytest.mark.asyncio
async def test_parse_goal_returns_valid_plan():
    parser = IntentParser()
    parser.groq_client = None
    mock_response = MagicMock()
    mock_response.text = '{"goal": "do something", "steps": ["test step"], "context": {}}'
    
    with patch("asyncio.to_thread", return_value=mock_response):
        plan = await parser.parse_goal("do something", "task-1")
        assert plan.goal == "do something"
        assert plan.task_id == "task-1"
        assert len(plan.steps) == 1
        assert plan.steps[0] == "test step"

@pytest.mark.asyncio
async def test_parse_response_strips_markdown():
    parser = IntentParser()
    raw = "```json\n{\"steps\": [], \"context\": {}}\n```"
    plan = parser._parse_response(raw, "test")
    assert plan.goal == "test"
    assert plan.steps == []

def test_semantic_mapper_finds_best_node():
    mapper = SemanticMapper()
    nodes = [
        NodeData(nodeId="1", className="Button", text="Submit"),
        NodeData(nodeId="2", className="Button", text="Cancel")
    ]
    
    # Mock embeddings to return high similarity for "Submit"
    mock_emb_response = MagicMock()
    # Assuming 2 nodes, 2 embeddings. Let's mock the values so Submit is closer.
    mock_emb_response.embeddings = [
        MagicMock(values=[1.0, 0.0]), # Submit
        MagicMock(values=[0.0, 1.0])  # Cancel
    ]
    
    with patch.object(mapper.client.models, "embed_content", return_value=mock_emb_response):
        with patch.object(mapper, "_embed", return_value=[0.9, 0.1]):
            best = mapper.find_best_node("click submit", nodes)
            assert best.text == "Submit"

def test_semantic_mapper_handles_empty_nodes():
    mapper = SemanticMapper()
    assert mapper.find_best_node("test", []) is None
