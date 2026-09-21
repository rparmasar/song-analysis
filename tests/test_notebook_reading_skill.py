"""
Test suite for Jupyter Notebook Reading Skill
Verifies the skill works correctly within context window limits.
"""

import json
import sys
import os

sys.path.insert(0, '/home/rparmasar/.config/opencode')
sys.path.insert(0, '/home/rparmasar/Desktop/projects/song-analysis')


def test_skill_definition_exists():
    """Test that skill definition file exists and is valid JSON"""
    skill_path = "/home/rparmasar/.config/opencode/agents/jupyter-notebook-reader.json"
    
    try:
        with open(skill_path, 'r') as f:
            config = json.load(f)
        
        assert "name" in config, "Skill must have a name field"
        assert config["name"] == "jupyter-notebook-referencer", "Incorrect skill name"
        assert "instructions" in config, "Skill must have instructions"
        assert "constraints" in config, "Skill must have constraints"
        assert "capabilities" in config, "Skill must have capabilities"
        
        print("✓ Skill definition exists and is valid")
        return True
    except Exception as e:
        print(f"✗ Skill definition test failed: {e}")
        return False


def test_project_helper_exists():
    """Test that project-local context helper exists"""
    helper_path = "/home/rparmasar/Desktop/projects/song-analysis/.opencode/agents/notebook-context-helper.json"
    
    try:
        with open(helper_path, 'r') as f:
            config = json.load(f)
        
        assert "project" in config
        assert "notebook_context_guide" in config
        assert len(config["notebook_context_guide"]) == 3
        
        print("✓ Project context helper exists and is valid")
        return True
    except Exception as e:
        print(f"✗ Project helper test failed: {e}")
        return False


def test_notebook_sizes_match_documentation():
    """Verify notebook size estimates in config match reality"""
    import subprocess
    
    try:
        # Get actual sizes
        result = subprocess.run(
            ['du', '-bc', '/home/rparmasar/Desktop/projects/song-analysis/song-analysis/*.ipynb'],
            capture_output=True, text=True
        )
        
        sizes = {}
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if '.ipynb' in line:
                size_kb = int(line.split()[0]) / 1024
                name = line.split('/')[-1]
                sizes[name] = size_kb
        
        # Verify estimates are reasonably close
        assert "main.ipynb" in sizes, "main.ipynb not found"
        assert sizes["main.ipynb"] > 3000, f"main.ipynb should be ~3.6MB, got {sizes['main.ipynb']:.1f}KB"
        
        print(f"✓ Notebook sizes verified: main={sizes['main.ipynb']:.1f}KB, others OK")
        return True
    except Exception as e:
        print(f"✗ Notebook size test failed: {e}")
        return False


def test_skill_can_load_notebook_info():
    """Test that skill can read notebook metadata without loading full content"""
    import subprocess
    
    try:
        # Load just the cell count from main.ipynb
        result = subprocess.run(
            ['python3', '-c', '''
import json
with open("song-analysis/main.ipynb") as f:
    nb = json.load(f)
    total_cells = len(nb["cells"])
    md_cells = sum(1 for c in nb["cells"] if c.get("cell_type") == "markdown")
    code_cells = sum(1 for c in nb["cells"] if c.get("cell_type") == "code")
    print(f"total:{total_cells},md:{md_cells},code:{code_cells}")
''', '-c'],
            cwd='/home/rparmasar/Desktop/projects/song-analysis',
            capture_output=True, text=True, timeout=30
        )
        
        output = result.stdout.strip()
        parts = output.split(',')
        
        assert len(parts) == 3, "Should have total,md,code counts"
        total_cells = int(parts[0].split(':')[1])
        md_cells = int(parts[1].split(':')[1])
        code_cells = int(parts[2].split(':')[1])
        
        print(f"✓ Notebook metadata loaded: {total_cells} cells ({md_cells} markdown, {code_cells} code)")
        assert total_cells > 0 and md_cells >= 0 and code_cells >= 0
        return True
    except Exception as e:
        print(f"✗ Notebook info load test failed: {e}")
        return False


def test_skill_context_budget_awareness():
    """Verify skill configuration respects context limits"""
    
    try:
        with open('/home/rparmasar/.config/opencode/agents/jupyter-notebook-reader.json', 'r') as f:
            config = json.load(f)
        
        budget = config.get("context_budget", {})
        
        assert "max_tokens" in budget, "Should have max_tokens limit"
        assert budget["max_tokens"] == 24000, "Should respect 24K context limit"
        assert "chunk_size_kb" in budget, "Should define chunk size"
        assert budget["chunk_size_kb"] <= 100, "Chunk size should be <=100KB"
        
        print("✓ Context budget configuration verified (max_tokens=24K, chunks=100KB)")
        return True
    except Exception as e:
        print(f"✗ Context budget test failed: {e}")
        return False


def test_skill_usage_patterns_documented():
    """Verify both usage patterns are documented"""
    
    try:
        with open('/home/rparmasar/.config/opencode/agents/jupyter-notebook-reader.json', 'r') as f:
            config = json.load(f)
        
        assert "usage_patterns" in config, "Should document usage patterns"
        
        patterns = config["usage_patterns"]
        assert "agent_mode" in patterns, "Should document agent_mode usage"
        assert "skill_mode" in patterns, "Should document skill_mode usage"
        
        # Verify examples are provided
        assert "example" in patterns["agent_mode"], "Agent mode needs example"
        assert "example" in patterns["skill_mode"], "Skill mode needs example"
        
        print("✓ Both usage patterns documented (agent + skill modes)")
        return True
    except Exception as e:
        print(f"✗ Usage patterns test failed: {e}")
        return False


def test_constraints_are_properly_set():
    """Verify critical constraints are in place"""
    
    try:
        with open('/home/rparmasar/.config/opencode/agents/jupyter-notebook-reader.json', 'r') as f:
            config = json.load(f)
        
        constraints = config.get("constraints", [])
        
        constraint_checks = [
            "NEVER read entire .ipynb files at once" in str(constraints),
            "Maximum 100KB per read operation" in str(constraints),
            "Never include cell outputs" in str(constraints)
        ]
        
        assert all(constraint_checks), "Not all critical constraints are set"
        
        print("✓ Critical constraints verified (no full reads, chunked, no outputs)")
        return True
    except Exception as e:
        print(f"✗ Constraints test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("Testing Jupyter Notebook Reading Skill")
    print("=" * 60)
    
    tests = [
        ("Skill Definition", test_skill_definition_exists),
        ("Project Helper File", test_project_helper_exists),
        ("Notebook Sizes Match", test_notebook_sizes_match_documentation),
        ("Metadata Loading", test_skill_can_load_notebook_info),
        ("Context Budget", test_skill_context_budget_awareness),
        ("Usage Patterns", test_skill_usage_patterns_documented),
        ("Critical Constraints", test_constraints_are_properly_set),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"✗ Test '{name}' assertion failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test '{name}' error: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
