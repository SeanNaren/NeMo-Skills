# Method

## Problem Formulation

We introduce a multi-turn interaction system for automated problem localization in software repositories, specifically designed for the SWE-bench benchmark. Given a natural language problem description and a target repository, our objective is to identify the precise code locations requiring modification to resolve the described issue. This task presents unique challenges including navigating large codebases, managing limited context windows, and maintaining focused exploration strategies across multiple reasoning steps.

## Multi-Turn Interaction Framework

Our approach models problem localization as a sequential decision-making process where an autonomous agent iteratively explores the codebase through structured tool interactions. The agent operates within a dialogue framework, maintaining a conversation history that captures both its reasoning process and the information gathered from the repository.

### Agent Architecture

The agent follows a perception-reasoning-action cycle:

1. **Perception**: The agent receives the current state comprising the problem description, accumulated tool outputs, and dialogue history.

2. **Reasoning**: Using a large language model, the agent analyzes available information to determine the next exploration action.

3. **Action**: The agent executes one of four available tools to gather additional information about the codebase.

### Available Tools

We provide four specialized tools that enable comprehensive repository exploration:

- **File Viewing**: Retrieves source code content with optional line range specification, enabling focused examination of specific code sections.

- **Repository Structure**: Generates a hierarchical view of the codebase, filtered to show only relevant source files while excluding non-essential directories.

- **Code Search**: Performs keyword-based search across the repository, returning contextual snippets around matches to facilitate understanding of code usage patterns.

- **Dependency Analysis**: Constructs import relationship graphs to understand module interconnections and identify related components.

### Exploration Strategy

The agent begins each problem-solving session by examining the repository structure to understand the codebase organization. It then employs a combination of targeted searches and file examinations based on keywords extracted from the problem description. The multi-turn nature allows the agent to refine its search strategy based on accumulated evidence, progressively narrowing the search space until identifying the relevant code locations.

## Context Management

A critical challenge in multi-turn interaction systems is managing the growing dialogue history within language model context limitations. We developed a comprehensive context management strategy with multiple components:

### Token Estimation

We implement context-aware token estimation that accounts for the syntactic differences between code and natural language. Code segments typically exhibit higher token density due to operators, keywords, and formatting, requiring approximately 3.5 characters per token, while natural language averages 4.5 characters per token. This differentiated estimation enables accurate context utilization planning.

### Truncation Strategies

When dialogue history approaches context limits, we employ one of several truncation strategies:

**Sequential Truncation**: Progressively removes the oldest interaction turns while preserving recent context. This strategy maintains conversation continuity and is suitable for problems requiring detailed tracking of recent explorations.

**Bookend Truncation**: Retains the initial problem statement and most recent interactions while removing intermediate turns. This approach ensures the agent maintains problem awareness while focusing on current exploration state, proving particularly effective for complex repositories where middle interactions often contain redundant information.

**Adaptive Truncation**: Dynamically selects between strategies based on conversation characteristics, with multi-level fallbacks for extreme cases requiring aggressive context reduction.

### Proactive Context Monitoring

Rather than reacting to context overflow errors, our system proactively monitors token usage before each interaction. This preemptive approach calculates expected token consumption and applies appropriate truncation strategies, preventing runtime failures and ensuring smooth operation throughout the exploration process.

## Loop Detection and Intervention

Multi-turn systems risk entering repetitive behavior patterns where agents repeatedly execute identical or similar actions without making progress. We address this through a comprehensive loop detection and intervention mechanism:

### Detection Mechanism

The system continuously monitors the agent's tool usage patterns, maintaining a sliding window of recent actions. When consecutive identical tool calls exceed a configurable threshold, the system identifies a potential behavioral loop. This detection operates at the semantic level, recognizing functionally equivalent calls even with minor parameter variations.

### Intervention Strategy

Upon detecting repetitive behavior, the system injects carefully crafted intervention messages into the dialogue history. These messages acknowledge the repetition and suggest alternative exploration strategies, effectively guiding the agent toward more productive actions. The interventions are designed to maintain the natural flow of conversation while providing actionable guidance.

### Pattern Analysis

Beyond immediate intervention, the system logs and analyzes repetitive patterns to identify common failure modes. This analysis reveals systematic biases in agent behavior and informs both prompt engineering improvements and architectural enhancements.

## Repository Filtering and Scoping

Efficient repository exploration requires intelligent filtering to focus on relevant code while excluding non-essential files:

### File Type Selection

We restrict exploration to source code files most likely to contain implementation logic. For Python repositories, this includes Python source files and configuration files that may contain settings affecting program behavior. This focused approach significantly reduces the search space while maintaining high recall for relevant code locations.

### Directory Filtering

We implement hierarchical filtering that excludes directories unlikely to contain production code. This includes:

- Testing infrastructure (test suites, fixtures)
- Build artifacts and caches
- Documentation and asset directories
- Development environment files
- Third-party dependencies

### Adaptive Filtering

Based on empirical analysis, we identified several directory types commonly excluded but frequently containing relevant code. Our system adaptively preserves:

- Utility modules that often implement core functionality
- Library directories that may represent primary source code
- Migration files that frequently contain bug fixes
- Configuration directories that may harbor erroneous settings

## Final Turn Optimization

To maximize success rates given finite interaction budgets, we implement a final turn optimization mechanism. When approaching the maximum allowed interactions, the system modifies its behavior to ensure productive output:

### Forced Prediction

On the final interaction turn, the system injects specialized instructions requiring the agent to synthesize all accumulated information into concrete location predictions. This prevents scenarios where exploration continues without producing actionable results.

### Instruction Alignment

The injected instructions maintain stylistic and structural consistency with the primary system prompt, ensuring the agent responds appropriately without confusion. Multiple instruction variants enable experimentation with different urgency levels and framing approaches.

### Graceful Degradation

This mechanism transforms potential failures due to interaction limits into partial successes. Even incomplete explorations yield location predictions based on accumulated evidence, providing value even when comprehensive analysis isn't achieved.

## Evaluation Framework

We implement comprehensive evaluation mechanisms to assess agent performance:

### Ground Truth Extraction

We parse repository patches to extract precise change locations, handling various edge cases including file creation, deletion, and modification. This ground truth serves as the definitive reference for accuracy assessment.

### Multi-Level Metrics

Our evaluation encompasses multiple granularity levels:
- **File-Level Accuracy**: Measures the agent's ability to identify correct files
- **Line-Level Precision**: Evaluates the specificity of location predictions
- **Coverage Analysis**: Assesses what proportion of ground truth locations were accessible given filtering constraints

### Failure Analysis

We categorize prediction failures to identify systematic issues:
- Files excluded by type filtering
- Files in filtered directories  
- Files absent from the repository version
- Locations correctly identified but imprecisely specified

## Robustness and Error Recovery

Real-world deployment requires handling various failure modes gracefully:

### Format Flexibility

The system accepts multiple tool invocation formats, including properly structured commands, partial specifications, and implicit requests. This flexibility accommodates variations in language model outputs while maintaining functional correctness.

### Progressive Degradation

When encountering errors, the system attempts multiple recovery strategies before failing:
- Suggesting similar files when exact matches aren't found
- Providing informative error messages with actionable suggestions
- Defaulting to broader searches when specific queries fail
- Maintaining operation despite individual tool failures

### Diagnostic Information

Throughout execution, the system generates detailed diagnostic logs enabling post-hoc analysis of agent behavior, context utilization, and failure modes. This instrumentation proves invaluable for system improvement and debugging.

## Conclusion

Our multi-turn interaction system addresses key challenges in automated problem localization through careful engineering of context management, behavioral control, and error recovery mechanisms. The modular design enables controlled experimentation while maintaining robustness across diverse repository structures and problem types. Comprehensive evaluation demonstrates the effectiveness of our approach on the challenging SWE-bench benchmark, with particular improvements in handling large repositories and complex multi-file modifications.
