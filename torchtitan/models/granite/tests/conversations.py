"""Canonical conversation fixtures for the granite test suite."""

SINGLE_TURN_REASONING = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4", "reasoning_content": "Simple addition."},
]

SINGLE_TURN_NO_REASONING = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help?"},
]

MULTI_TURN_ALL_REASONING = [
    {"role": "user", "content": "Q0"},
    {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
]

MULTI_TURN_NO_REASONING = [
    {"role": "user", "content": "Q0"},
    {"role": "assistant", "content": "A0"},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1"},
]

MULTI_TURN_MIXED_REASONING = [
    {"role": "user", "content": "Q0"},
    {"role": "assistant", "content": "A0"},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
]

TOOL_CHAIN_REASONING = [
    {"role": "user", "content": "Search for X"},
    {"role": "assistant", "content": "call_tool", "reasoning_content": "Need to search"},
    {"role": "tool", "content": "result of search"},
    {"role": "assistant", "content": "Found it", "reasoning_content": "Got result"},
    {"role": "user", "content": "Thanks"},
    {"role": "assistant", "content": "Welcome", "reasoning_content": "Done"},
]

TOOL_CHAIN_NO_REASONING = [
    {"role": "user", "content": "Search for X"},
    {"role": "assistant", "content": "call_tool"},
    {"role": "tool", "content": "result of search"},
    {"role": "assistant", "content": "Found it"},
    {"role": "user", "content": "Thanks"},
    {"role": "assistant", "content": "Welcome", "reasoning_content": "Done"},
]

MANY_TURNS_MIXED = [
    {"role": "user", "content": "Q0"},
    {"role": "assistant", "content": "A0", "reasoning_content": "R0"},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1"},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
    {"role": "user", "content": "Q3"},
    {"role": "assistant", "content": "A3"},
    {"role": "user", "content": "Q4"},
    {"role": "assistant", "content": "A4", "reasoning_content": "R4"},
]

WITH_SYSTEM_MESSAGE = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Q0"},
    {"role": "assistant", "content": "A0"},
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
]

TOOL_CHAIN_NO_TRAILING_USER = [
    {"role": "user", "content": "Search for X"},
    {"role": "assistant", "content": "call_tool", "reasoning_content": "Need search"},
    {"role": "tool", "content": "result of search"},
    {"role": "assistant", "content": "Here's what I found", "reasoning_content": "Got it"},
]

MULTI_STEP_AGENT_NO_TRAILING_USER = [
    {"role": "user", "content": "Fix the bug"},
    {"role": "assistant", "content": "read file", "reasoning_content": "R0"},
    {"role": "tool", "content": "file content"},
    {"role": "assistant", "content": "edit file", "reasoning_content": "R1"},
    {"role": "tool", "content": "ok"},
    {"role": "assistant", "content": "Done", "reasoning_content": "R2"},
]

NO_REASONING_TOOL_CHAIN_NO_TRAILING_USER = [
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "calc(2+2)"},
    {"role": "tool", "content": "4"},
    {"role": "assistant", "content": "The answer is 4"},
]

MIXED_REASONING_NO_TRAILING_USER = [
    {"role": "user", "content": "Help me"},
    {"role": "assistant", "content": "step1"},
    {"role": "tool", "content": "result1"},
    {"role": "assistant", "content": "step2", "reasoning_content": "thinking"},
    {"role": "tool", "content": "result2"},
    {"role": "assistant", "content": "final answer", "reasoning_content": "done"},
]

ALL_CONVERSATIONS = [
    ("single_turn_reasoning", SINGLE_TURN_REASONING),
    ("single_turn_no_reasoning", SINGLE_TURN_NO_REASONING),
    ("multi_turn_all_reasoning", MULTI_TURN_ALL_REASONING),
    ("multi_turn_no_reasoning", MULTI_TURN_NO_REASONING),
    ("multi_turn_mixed_reasoning", MULTI_TURN_MIXED_REASONING),
    ("tool_chain_reasoning", TOOL_CHAIN_REASONING),
    ("tool_chain_no_reasoning", TOOL_CHAIN_NO_REASONING),
    ("many_turns_mixed", MANY_TURNS_MIXED),
    ("with_system_message", WITH_SYSTEM_MESSAGE),
    ("tool_chain_no_trailing_user", TOOL_CHAIN_NO_TRAILING_USER),
    ("multi_step_agent_no_trailing_user", MULTI_STEP_AGENT_NO_TRAILING_USER),
    ("no_reasoning_tool_chain_no_trailing_user", NO_REASONING_TOOL_CHAIN_NO_TRAILING_USER),
    ("mixed_reasoning_no_trailing_user", MIXED_REASONING_NO_TRAILING_USER),
]
