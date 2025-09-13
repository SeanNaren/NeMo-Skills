# Loop Detection Analysis

## The Problem

The agent is getting stuck in a repetitive loop where it generates the same tool call over and over:

```json
{"view_file": {"path": "sympy/polys/polytools.py", "view_range": [1, -1]}}
```

This was repeated 18 out of 20 times in the example!

## Root Cause

1. **Large File Truncation**: The file has 7152 lines, but gets truncated
2. **Lost Context**: The agent doesn't realize the output was truncated
3. **No Memory**: The agent doesn't remember it already tried this
4. **Bookend Truncation**: With bookend truncation, the agent loses the middle context showing previous attempts

## Why This Happens with Bookend Truncation

Bookend truncation keeps:
- First turn (problem statement)
- Last 1-2 turns

But when stuck in a loop:
- The "last turns" are all the same failed attempt
- The agent doesn't see its previous attempts
- It keeps trying the same thing

## Solutions

### 1. Loop Detection (Immediate Fix)
Add logic to detect when the agent generates the same tool call multiple times:
- Track last N tool calls
- If same call appears 3+ times, intervene
- Either skip the generation or modify the prompt

### 2. Tool Call History in Context
Include a summary of recent tool calls in the agent's context:
```
Recent tool calls:
- view_file(sympy/polys/polytools.py, [1, -1]) - tried 5 times, output truncated
```

### 3. Smart Truncation for Loops
When a loop is detected, modify bookend truncation to include:
- First turn (problem)
- One example of the repeated attempt
- A system message about the loop
- Last different turn before the loop started

### 4. Intervention Messages
When loop detected, inject a system message:
```
SYSTEM: You have tried view_file on this file 5 times. The file is too large (7152 lines). 
Try viewing a specific section or using a different approach.
```

### 5. Tool Output Enhancement
Make truncation explicit in tool outputs:
```
File: sympy/polys/polytools.py (lines 1-1000) [TRUNCATED - Original file has 7152 total lines]
WARNING: File content truncated. Use specific line ranges to view sections.
```

## Implementation Priority

1. **Loop Detection** - Prevent infinite loops (HIGH)
2. **Intervention Messages** - Guide the agent out of loops (HIGH)
3. **Tool Output Enhancement** - Make truncation clear (MEDIUM)
4. **Smart Loop-Aware Truncation** - Better context management (LOW)
