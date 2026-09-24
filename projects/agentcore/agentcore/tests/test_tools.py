"""Registry validation, the AST calculator, and the sandboxed read_note."""
import pytest

from agentcore.tools import (
    Registry,
    Tool,
    ToolError,
    calculator_tool,
    read_note_tool,
    text_stats_tool,
)


# -- calculator -------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2+2*3", 8),                    # precedence
        ("(2+8)/5", 2.0),                # division, parentheses
        ("10/4", 2.5),                   # true division
        ("10//3", 3),                    # floor division
        ("7 % 3", 1),                    # modulo
        ("2**10", 1024),                 # power
        ("-3+5", 2),                     # unary minus
        ("2*(3+4)-6/3", 12.0),           # compound
    ],
)
def test_calculator_correctness(expression, expected):
    assert calculator_tool().handler(expression=expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",  # the classic eval payload
        "().__class__.__bases__",
        "1 + x",                          # unknown name
        "'2' + '2'",                      # strings
        "True + 1",                       # booleans are not numbers here
        "9**9**9",                        # exponent blowup
        "1 +",                            # syntax error
        "",                               # empty
    ],
)
def test_calculator_rejects_everything_but_arithmetic(expression):
    tool = calculator_tool()
    with pytest.raises(ToolError):
        tool.handler(expression=expression)


def test_calculator_division_by_zero_is_toolerror():
    with pytest.raises(ToolError, match="division by zero"):
        calculator_tool().handler(expression="1/0")


# -- registry validation ------------------------------------------------------


def test_registry_type_validation():
    reg = Registry.with_builtins()
    with pytest.raises(ToolError, match="must be string, got int"):
        reg.call("calculator", {"expression": 123})
    with pytest.raises(ToolError, match="must be string, got float"):
        reg.call("calculator", {"expression": 1.5})
    with pytest.raises(ToolError, match="must be string, got bool"):
        reg.call("text_stats", {"text": True})
    assert reg.call("text_stats", {"text": "hello world"}) == {"words": 2, "chars": 11, "lines": 1}


def test_registry_unknown_and_missing_and_extra():
    reg = Registry.with_builtins()
    with pytest.raises(ToolError, match="unknown tool"):
        reg.call("does_not_exist", {})
    with pytest.raises(ToolError, match="missing required argument"):
        reg.call("calculator", {})
    with pytest.raises(ToolError, match="unexpected argument"):
        reg.call("calculator", {"expression": "1+1", "evil": True})


def test_schema_shape_and_listing():
    reg = Registry.with_builtins()
    schema = reg.schema("calculator")
    assert schema["name"] == "calculator"
    assert schema["description"]
    assert schema["parameters"]["properties"]["expression"] == {"type": "string"}
    assert schema["parameters"]["required"] == ["expression"]
    names = [t.name for t in reg.list()]
    assert names == ["calculator", "text_stats"]  # read_note only joins with a sandbox
    assert "dangerous" in schema
    with pytest.raises(ToolError, match="unknown tool"):
        reg.schema("nope")


def test_registry_refuses_bad_tool_definitions():
    reg = Registry()
    with pytest.raises(ToolError, match="allowed types"):
        reg.register(Tool("bad", "d", {"x": list}, lambda x: x))  # type: ignore[arg-type]
    with pytest.raises(ToolError, match="already registered"):
        reg.register(Tool("dup", "d", {}, lambda: 1))
        reg.register(Tool("dup", "d", {}, lambda: 2))


# -- read_note sandbox --------------------------------------------------------


@pytest.fixture()
def sandbox(tmp_path):
    (tmp_path / "note.txt").write_text("hello from the sandbox", encoding="utf-8")
    sub = tmp_path / "notes"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested", encoding="utf-8")
    return tmp_path


def test_read_note_within_sandbox(sandbox):
    tool = read_note_tool(sandbox)
    assert tool.handler(path="note.txt") == "hello from the sandbox"
    assert tool.handler(path="notes/nested.txt") == "nested"
    assert tool.dangerous is True


@pytest.mark.parametrize(
    "bad_path",
    [
        "../secrets.txt",        # plain traversal
        "notes/../../escape.txt",  # traversal mid-path
        "/etc/passwd",           # POSIX absolute
        "C:\\x",                 # Windows absolute / backslash
        "..\\windows.txt",       # backslash traversal
        "\\\\server\\share",     # UNC path
        ".",                     # the sandbox itself is not a file
        "",                      # empty
        "missing.txt",           # just not there
    ],
)
def test_sandbox_escape_attempts_raise(sandbox, bad_path):
    tool = read_note_tool(sandbox)
    with pytest.raises(ToolError):
        tool.handler(path=bad_path)


def test_read_note_refused_without_confirm_allowed_with(sandbox):
    reg = Registry.with_builtins(sandbox)
    with pytest.raises(ToolError, match="refused"):
        reg.call("read_note", {"path": "note.txt"})  # no confirmation given
    with pytest.raises(ToolError, match="refused"):
        reg.call("read_note", {"path": "note.txt"}, confirm_dangerous=lambda name: False)
    asked = []
    value = reg.call(
        "read_note",
        {"path": "note.txt"},
        confirm_dangerous=lambda name: asked.append(name) or True,
    )
    assert value == "hello from the sandbox"
    assert asked == ["read_note"]  # the callback receives the tool name


def test_handler_crash_is_wrapped_as_toolerror():
    def boom(x: str) -> str:
        raise ValueError("kaboom")

    reg = Registry()
    reg.register(Tool("boom", "always fails", {"x": str}, boom))
    with pytest.raises(ToolError, match="kaboom"):
        reg.call("boom", {"x": "1"})
