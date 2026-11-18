import random

import pytest

from mini_agent.history import normalize, repair, validate
from mini_agent.messages import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


def use(id: str, name: str = "read_file", **inp) -> ToolUseBlock:
    return ToolUseBlock(id=id, name=name, input=inp)


def result(id: str, content: str = "ok", is_error: bool = False) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=id, content=content, is_error=is_error)


def asst(*blocks) -> Message:
    return Message(role="assistant", content=list(blocks))


def usr(*blocks) -> Message:
    return Message(role="user", content=list(blocks))


def codes(messages) -> list[str]:
    return [v.code for v in validate(messages)]


def valid_history() -> list[Message]:
    return [
        Message.user("fix the bug"),
        asst(TextBlock(text="looking"), use("a"), use("b")),
        usr(result("a"), result("b")),
        Message.assistant("done"),
    ]


# ---------- validate: one violation per invariant ----------


def test_valid_history_has_no_violations():
    assert validate(valid_history()) == []
    assert validate([]) == []


def test_i1_dangling_tool_use_at_end():
    msgs = [Message.user("hi"), asst(use("a"))]
    assert codes(msgs) == ["I1"]


def test_i1_missing_one_of_several_results():
    msgs = [Message.user("hi"), asst(use("a"), use("b"), use("c")), usr(result("a"), result("c"))]
    violations = validate(msgs)
    assert [v.code for v in violations] == ["I1"]
    assert "'b'" in violations[0].detail


def test_i1_duplicate_result():
    msgs = [Message.user("hi"), asst(use("a")), usr(result("a"), result("a"))]
    assert codes(msgs) == ["I1"]


def test_i1_tool_use_followed_by_assistant_message():
    msgs = [Message.user("hi"), asst(use("a")), Message.assistant("oops")]
    assert "I1" in codes(msgs)


def test_i2_orphan_result_without_any_tool_use():
    msgs = [Message.user("hi"), Message.assistant("hello"), usr(result("zzz"))]
    assert codes(msgs) == ["I2"]


def test_i2_result_with_wrong_id():
    msgs = [Message.user("hi"), asst(use("a")), usr(result("a"), result("other"))]
    assert codes(msgs) == ["I2"]


def test_i3_first_message_from_assistant():
    assert codes([Message.assistant("hi")]) == ["I3"]


def test_i3_consecutive_same_role():
    assert codes([Message.user("a"), Message.user("b")]) == ["I3"]
    assert codes([Message.user("a"), Message.assistant("b"), Message.assistant("c")]) == ["I3"]


def test_i4_duplicate_tool_use_id_across_turns():
    msgs = [
        Message.user("hi"),
        asst(use("a")),
        usr(result("a")),
        asst(use("a")),
        usr(result("a")),
    ]
    assert "I4" in codes(msgs)


def test_i5_empty_content_and_blank_text():
    assert "I5" in codes([Message(role="user", content=[])])
    assert "I5" in codes([Message.user("   ")])


def test_role_block_mismatch_is_rejected_at_construction():
    with pytest.raises(ValueError):
        Message(role="assistant", content=[result("a")])
    with pytest.raises(ValueError):
        Message(role="user", content=[use("a")])


# ---------- repair ----------


def test_repair_leaves_valid_history_unchanged():
    msgs = valid_history()
    assert repair(msgs) == msgs


def test_repair_adds_synthetic_results_for_dangling_tool_uses():
    msgs = [Message.user("hi"), asst(use("a"), use("b"), use("c")), usr(result("b", "real"))]
    fixed = repair(msgs, missing_result_text="cancelled")
    assert validate(fixed) == []
    results = {r.tool_use_id: r for r in fixed[2].tool_results}
    assert results["b"].content == "real" and not results["b"].is_error
    assert results["a"].content == "cancelled" and results["a"].is_error
    assert results["c"].content == "cancelled" and results["c"].is_error


def test_repair_dangling_tool_use_at_end_gets_user_message():
    fixed = repair([Message.user("hi"), asst(use("a"))])
    assert validate(fixed) == []
    assert fixed[-1].role == "user" and fixed[-1].tool_results[0].tool_use_id == "a"


def test_repair_drops_orphan_and_duplicate_results():
    msgs = [Message.user("hi"), asst(use("a")), usr(result("a", "first"), result("a", "second"), result("zzz"))]
    fixed = repair(msgs)
    assert validate(fixed) == []
    assert [(r.tool_use_id, r.content) for r in fixed[2].tool_results] == [("a", "first")]


def test_repair_handles_orphan_result_message_between_assistants():
    msgs = [Message.user("hi"), Message.assistant("one"), usr(result("zzz")), Message.assistant("two")]
    fixed = repair(msgs)
    assert validate(fixed) == []
    assert len(fixed) == 2 and fixed[1].text == "onetwo"


def test_repair_merges_assistant_after_tool_use_and_answers_the_call():
    msgs = [Message.user("hi"), asst(use("a")), Message.assistant("next")]
    fixed = repair(msgs)
    assert validate(fixed) == []
    assert [m.role for m in fixed] == ["user", "assistant", "user"]
    assert fixed[2].tool_results[0].tool_use_id == "a"


def test_repair_drops_duplicate_tool_use_ids():
    msgs = [Message.user("hi"), asst(use("a")), usr(result("a")), asst(use("a")), usr(result("a"))]
    fixed = repair(msgs)
    assert validate(fixed) == []


def test_repair_prepends_placeholder_when_history_starts_with_assistant():
    fixed = repair([Message.assistant("hi"), Message.user("yo")])
    assert validate(fixed) == []
    assert fixed[0].role == "user"


def test_repair_removes_empty_messages_and_blank_text():
    msgs = [Message.user("hi"), Message(role="assistant", content=[]), Message.user("  "), Message.assistant("x")]
    fixed = repair(msgs)
    assert validate(fixed) == []
    assert [m.text for m in fixed] == ["hi", "x"]


# ---------- normalize ----------


def test_normalize_merges_same_role_and_puts_results_first():
    msgs = [
        Message.user("hi"),
        asst(use("a")),
        Message.user("extra note"),
        usr(result("a")),
    ]
    normalized = normalize(msgs)
    assert [m.role for m in normalized] == ["user", "assistant", "user"]
    blocks = normalized[2].content
    assert isinstance(blocks[0], ToolResultBlock) and isinstance(blocks[1], TextBlock)
    assert validate(normalized) == []


def test_normalize_does_not_mutate_input():
    msgs = [Message.user("a"), Message.user("b")]
    normalize(msgs)
    assert len(msgs) == 2


# ---------- randomized: repair must always produce a valid history ----------


def _random_message(rng: random.Random) -> Message:
    ids = ["a", "b", "c", "d"]
    if rng.random() < 0.5:
        blocks = []
        for _ in range(rng.randint(0, 3)):
            if rng.random() < 0.3:
                blocks.append(TextBlock(text=rng.choice(["", " ", "hello"])))
            else:
                blocks.append(use(rng.choice(ids)))
        return asst(*blocks)
    blocks = []
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.5:
            blocks.append(TextBlock(text=rng.choice(["", "question", "note"])))
        else:
            blocks.append(result(rng.choice(ids)))
    return usr(*blocks)


@pytest.mark.parametrize("seed", range(300))
def test_repair_always_yields_valid_history(seed):
    rng = random.Random(seed)
    msgs = [_random_message(rng) for _ in range(rng.randint(0, 8))]
    fixed = repair(msgs)
    assert validate(fixed) == [], (msgs, fixed)
    assert repair(fixed) == fixed


# ---------- messages ----------


def test_from_raw_valid_json_object():
    block = ToolUseBlock.from_raw("a", "read_file", '{"path": "x.py"}')
    assert block.input == {"path": "x.py"} and block.arguments_valid


@pytest.mark.parametrize("raw", ['{"path": "x', "[1, 2]", '"str"', "42"])
def test_from_raw_invalid_or_non_object_is_never_stored_as_input(raw):
    block = ToolUseBlock.from_raw("a", "read_file", raw)
    assert block.input == {} and block.raw_arguments == raw and not block.arguments_valid


def test_from_raw_empty_arguments_means_no_arguments():
    block = ToolUseBlock.from_raw("a", "list_files", "  ")
    assert block.input == {} and block.arguments_valid


def test_message_json_roundtrip_preserves_block_types():
    msgs = valid_history()
    restored = [Message.model_validate_json(m.model_dump_json()) for m in msgs]
    assert restored == msgs
    assert isinstance(restored[2].content[0], ToolResultBlock)


def test_usage_addition():
    total = Usage(input_tokens=10, output_tokens=2) + Usage(input_tokens=5, output_tokens=1, estimated=True)
    assert (total.input_tokens, total.output_tokens, total.estimated) == (15, 3, True)
