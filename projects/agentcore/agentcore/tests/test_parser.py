"""Wire-format parser: clean text, multiple blocks, and every malformed shape."""
import pytest

from agentcore.parser import ParsedToolCall, ParseError, parse_tool_calls


def test_clean_text_and_single_call():
    text = (
        "Let me compute that.\n"
        "```tool\n"
        '{"name": "calculator", "arguments": {"expression": "2+2*3"}}\n'
        "```\n"
        "One moment."
    )
    clean, calls = parse_tool_calls(text)
    assert clean == "Let me compute that.\n\nOne moment."
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call, ParsedToolCall)
    assert call.name == "calculator"
    assert call.arguments == {"expression": "2+2*3"}
    assert call.raw.startswith("```tool") and call.raw.endswith("```")


def test_multiple_calls_in_order_with_prose_between():
    text = (
        "First:\n```tool\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n```\n"
        "Then:\n```tool\n{\"name\": \"b\"}\n```\nDone."
    )
    clean, calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].arguments == {"x": 1}
    assert calls[1].arguments == {}  # arguments may be omitted
    assert "First:" in clean and "Then:" in clean and "Done." in clean
    assert "```" not in clean and '"name"' not in clean


def test_malformed_json_raises_with_reason():
    with pytest.raises(ParseError) as err:
        parse_tool_calls('```tool\n{"name": "calculator", "arguments": {"expression": 2+2}}\n```')
    assert "invalid JSON" in str(err.value)


def test_non_object_body_and_bad_name_raise():
    with pytest.raises(ParseError, match="JSON object"):
        parse_tool_calls("```tool\n[1, 2, 3]\n```")
    with pytest.raises(ParseError, match="missing a string 'name'"):
        parse_tool_calls("```tool\n{\"arguments\": {}}\n```")
    with pytest.raises(ParseError, match="arguments"):
        parse_tool_calls("```tool\n{\"name\": \"x\", \"arguments\": [1]}\n```")


def test_unterminated_block_raises():
    with pytest.raises(ParseError, match="unterminated"):
        parse_tool_calls("Working on it...\n```tool\n{\"name\": \"a\"}\n")
