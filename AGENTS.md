# AGENTS

<skills_system priority="1">

## Available Skills

<!-- SKILLS_TABLE_START -->
<usage>
When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke: `npx openskills read <skill-name>` (run in your shell)
  - For multiple: `npx openskills read skill-one,skill-two`
- The skill content will load with detailed instructions on how to complete the task
- Base directory provided in output for resolving bundled resources (references/, scripts/, assets/)

Usage notes:
- Only use skills listed in <available_skills> below
- Do not invoke a skill that is already loaded in your context
- Each skill invocation is stateless
</usage>

<available_skills>

<skill>
<name>filesystem</name>
<description>Direct filesystem operations (read, write, edit, list, search files and directories). Use for any file manipulation tasks including reading file contents, writing or overwriting files, editing/replacing text in files, copying, moving, deleting files, listing directories, building directory trees, and searching with glob patterns. Also use when the user asks to view, show, display, or inspect files or folder structures.</description>
<location>project</location>
</skill>

<skill>
<name>skill-creator</name>
<description>Guide for creating effective skills. Use when users want to create or update a skill that extends the agent's capabilities with specialized knowledge, workflows, or tool integrations.</description>
<location>project</location>
</skill>

<skill>
<name>terminal</name>
<description>Safe terminal operations using CAMEL TerminalToolkit utils (check_command_safety, sanitize_command, env setup helpers). Use for executing shell commands with safety preflight.</description>
<location>project</location>
</skill>

<skill>
<name>uv-pip-install</name>
<description>Install missing Python packages using uv pip. Use when (1) a Python import fails with ModuleNotFoundError, (2) user asks to install a Python package, (3) a script requires a missing dependency. This skill automatically detects missing libraries and installs them in the uv-managed .venv environment.</description>
<location>project</location>
</skill>

<skill>
<name>web-search</name>
<description>Web search and content fetching using Serper and crawl4ai. Use when the agent needs to search the web for information or fetch content from URLs.</description>
<location>project</location>
</skill>

</available_skills>
<!-- SKILLS_TABLE_END -->

</skills_system>
