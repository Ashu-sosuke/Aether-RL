import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from intent_parser import GeminiAccessDeniedError, IntentParser
from semantic_mapper import SemanticMapper
from websocket_server import _user_facing_error
from models import NodeData

@pytest.mark.asyncio
async def test_parse_goal_returns_valid_plan():
    parser = IntentParser()
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

@pytest.mark.asyncio
async def test_gemini_permission_denied_gets_clear_error():
    parser = IntentParser()
    with patch("asyncio.to_thread", side_effect=Exception("403 PERMISSION_DENIED. Your project has been denied access.")):
        with pytest.raises(GeminiAccessDeniedError) as exc:
            await parser._generate_json_with_fallback("prompt")

    assert "Gemini API access was denied" in str(exc.value)
    assert "GEMINI_API_KEY" in str(exc.value)

def test_websocket_keeps_known_llm_errors_user_facing():
    error = GeminiAccessDeniedError("Gemini API access was denied for the configured Google project/API key.")
    assert _user_facing_error(error) == str(error)

def test_websocket_prefixes_unknown_errors():
    assert _user_facing_error(RuntimeError("boom")) == "Server Error: boom"

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
