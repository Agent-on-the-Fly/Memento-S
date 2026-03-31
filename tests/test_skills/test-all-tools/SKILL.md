---
name: test_all_builtin_tools
description: A skill to test the invocation of all built-in tools via the SkillGateway.
execution_mode: function
---

```python
import asyncio
from builtin.tools.registry import execute_builtin_tool

async def main():
    print("--- Starting Built-in Tool Integration Test ---")
    
    test_dir = "test_skill_workspace"
    test_file = f"{test_dir}/test_file.txt"
    
    # 0. Cleanup from previous runs
    print("\n[Phase 0: Cleanup]")
    cleanup_result = await execute_builtin_tool("bash", {"command": f"rm -rf {test_dir}"})
    print(f"Cleanup result: {cleanup_result}")

    # 1. Test list_dir
    print("\n[Phase 1: Testing list_dir]")
    list_result = await execute_builtin_tool("list_dir", {"path": "."})
    print(f"list_dir result:\n{list_result}")
    assert "test_skills" in list_result, "list_dir failed to find test_skills dir"

    # 2. Test file_create
    print("\n[Phase 2: Testing file_create]")
    create_content = "Hello, World!\nThis is a test file."
    create_result = await execute_builtin_tool("file_create", {"path": test_file, "content": create_content})
    print(f"file_create result: {create_result}")
    assert "SUCCESS" in create_result, "file_create failed"

    # 3. Test read_file
    print("\n[Phase 3: Testing read_file]")
    read_result = await execute_builtin_tool("read_file", {"path": test_file})
    print(f"read_file result:\n{read_result}")
    assert "Hello, World!" in read_result, "read_file failed to read content"

    # 4. Test edit_file_by_lines
    print("\n[Phase 4: Testing edit_file_by_lines]")
    edit_content = "Hello, Memento-S!\nThis is an edited line."
    edit_result = await execute_builtin_tool("edit_file_by_lines", {"path": test_file, "start_line": 1, "end_line": 2, "new_content": edit_content})
    print(f"edit_file_by_lines result:\n{edit_result}")
    assert "SUCCESS" in edit_result and ">>" in edit_result, "edit_file_by_lines failed"

    # 5. Verify edit with read_file again
    print("\n[Phase 5: Verifying edit with read_file]")
    verify_read_result = await execute_builtin_tool("read_file", {"path": test_file})
    print(f"verify_read_result:\n{verify_read_result}")
    assert "Hello, Memento-S!" in verify_read_result, "File content was not edited correctly"

    # 6. Test search_grep
    print("\n[Phase 6: Testing search_grep]")
    grep_result = await execute_builtin_tool("search_grep", {"pattern": "Memento-S", "dir_path": test_dir})
    print(f"search_grep result:\n{grep_result}")
    assert "test_file.txt:1" in grep_result, "search_grep failed to find pattern"

    # 7. Test bash
    print("\n[Phase 7: Testing bash]")
    bash_result = await execute_builtin_tool("bash", {"command": "echo 'Hello from bash'"})
    print(f"bash result: {bash_result}")
    assert "Hello from bash" in bash_result, "bash tool failed"

    # 8. Test fetch_webpage
    print("\n[Phase 8: Testing fetch_webpage]")
    web_result = await execute_builtin_tool("fetch_webpage", {"url": "http://example.com"})
    print(f"fetch_webpage result (first 100 chars): {web_result[:100]}...")
    assert "Example Domain" in web_result, "fetch_webpage failed"
    
    print("\n--- Test Completed Successfully ---")

if __name__ == "__main__":
    asyncio.run(main())
```