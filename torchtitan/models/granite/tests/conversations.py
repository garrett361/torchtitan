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

# BackboneSuffix finds label-end boundaries by rendering a sub-conversation up to
# a given turn, measuring its character length, and using that as a position lookup
# into the full conversation's token offset table. But the chat template internally
# decides whether to truncate thinking based on the position of the last user
# message in the rendered slice. When the slice is shorter than the full
# conversation, the last-user position differs, the same turn may be rendered
# through a different template branch (one that trims leading whitespace), and
# the character lengths diverge. The result: the offset lookup overshoots,
# labeling tokens from the next turn's header.

LEADING_NEWLINE_NO_REASONING_BEFORE_REASONING = [
    {"role": "user", "content": "Translate 'hello' to French."},
    {"role": "assistant", "content": "\nBonjour."},
    {"role": "user", "content": "Now translate 'goodbye'."},
    {"role": "assistant", "content": "Au revoir.", "reasoning_content": "Simple translation."},
]

LEADING_DOUBLE_NEWLINE_TOOL_CHAIN = [
    {"role": "user", "content": "What's the current BTC price?"},
    {"role": "assistant", "content": "Let me look that up."},
    {"role": "tool", "content": '{"price_usd": 67432.10, "change_24h": "+2.3%"}'},
    {"role": "assistant", "content": "\n\nBitcoin is currently $67,432.10, up 2.3%."},
    {"role": "user", "content": "Should I buy?"},
    {"role": "assistant", "content": "I can't give financial advice.", "reasoning_content": "Must not provide investment recommendations."},
]

MULTIPLE_NO_REASONING_GROUPS_LEADING_WS = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "\n\nThe answer is 4."},
    {"role": "user", "content": "Now explain why."},
    {"role": "assistant", "content": "Addition is commutative.", "reasoning_content": "Thinking about commutativity."},
    {"role": "user", "content": "What about 3+3?"},
    {"role": "assistant", "content": "\n  The answer is 6."},
    {"role": "user", "content": "Summarize everything."},
    {"role": "assistant", "content": "We covered basic addition.", "reasoning_content": "Summarizing."},
]

ADJACENT_NO_REASONING_GROUPS_LEADING_WS = [
    {"role": "user", "content": "Say hello."},
    {"role": "assistant", "content": "\n\nHello there!"},
    {"role": "user", "content": "Say goodbye."},
    {"role": "assistant", "content": "\n  Goodbye!"},
    {"role": "user", "content": "Now think about life."},
    {"role": "assistant", "content": "Life is complex.", "reasoning_content": "Reflecting."},
]

MULTI_TURN_TOOL_CHAIN_MIDDLE_LEADING_WS = [
    {"role": "user", "content": "Find the weather and convert to Fahrenheit."},
    {"role": "assistant", "content": "Let me check the weather."},
    {"role": "tool", "content": '{"temp_c": 20, "condition": "sunny"}'},
    {"role": "assistant", "content": "\n\nNow converting 20C to Fahrenheit."},
    {"role": "tool", "content": '{"temp_f": 68}'},
    {"role": "assistant", "content": "\n It is 68F and sunny."},
    {"role": "user", "content": "Thanks, what about tomorrow?"},
    {"role": "assistant", "content": "No forecast data.", "reasoning_content": "No tool for forecasts."},
]

LEADING_TABS_AND_SPACES = [
    {"role": "user", "content": "Help me write a config."},
    {"role": "assistant", "content": "What format?", "reasoning_content": "Need to clarify."},
    {"role": "user", "content": "YAML please."},
    {"role": "assistant", "content": "\t  name: app\n\t  port: 8080"},
    {"role": "user", "content": "Add a host field."},
    {"role": "assistant", "content": "name: app\nport: 8080\nhost: localhost", "reasoning_content": "Adding host."},
]

REASONING_ONLY_WHITESPACE_WITH_LEADING_WS = [
    {"role": "user", "content": "Solve x^2=4."},
    {"role": "assistant", "content": "\n  x = 2 or x = -2", "reasoning_content": "  \n  "},
    {"role": "user", "content": "Show your work."},
    {"role": "assistant", "content": "x^2=4 means x=+-2.", "reasoning_content": "Factor or sqrt."},
]

CONTENT_WITH_IM_TAGS = [
    {"role": "user", "content": "Show me how <|im_start|> and <|im_end|> delimit messages."},
    {"role": "assistant", "content": "Each message is wrapped: <|im_start|>role\\ncontent<|im_end|>\\n", "reasoning_content": "Explaining chat template structure."},
    {"role": "user", "content": "Can <|im_end|> appear inside content?"},
    {"role": "assistant", "content": "Yes, but it will be tokenized as the special token, which may break parsing."},
]

# Conversations with <think>/</think> in assistant content are excluded: the
# template's heuristic (`'<think>' not in content`) skips adding structural
# <think></think> and the truncation path (`c.split('</think>')[-1]`) corrupts
# content. See TestChatTemplate in test_data.py for documentation of this behavior.
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
    ("leading_newline_no_reasoning_before_reasoning", LEADING_NEWLINE_NO_REASONING_BEFORE_REASONING),
    ("leading_double_newline_tool_chain", LEADING_DOUBLE_NEWLINE_TOOL_CHAIN),
    ("multiple_no_reasoning_groups_leading_ws", MULTIPLE_NO_REASONING_GROUPS_LEADING_WS),
    ("adjacent_no_reasoning_groups_leading_ws", ADJACENT_NO_REASONING_GROUPS_LEADING_WS),
    ("multi_turn_tool_chain_middle_leading_ws", MULTI_TURN_TOOL_CHAIN_MIDDLE_LEADING_WS),
    ("leading_tabs_and_spaces", LEADING_TABS_AND_SPACES),
    ("reasoning_only_whitespace_with_leading_ws", REASONING_ONLY_WHITESPACE_WITH_LEADING_WS),
    ("content_with_im_tags", CONTENT_WITH_IM_TAGS),
]
