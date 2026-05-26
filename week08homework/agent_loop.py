#!/usr/bin/env python3
import subprocess
import sys
import shutil
from pathlib import Path

_FILES_DIR = Path(__file__).resolve().parent.parent
if str(_FILES_DIR) not in sys.path:
    sys.path.insert(0, str(_FILES_DIR))

from gemini_simple_api import GeminiSimpleAPI

TASK_DIR = Path(__file__).parent
TEST_DIR = TASK_DIR / "tests"
TEST_FILE = TEST_DIR / "test_bayes_factor.py"
SOURCE_FILE = TASK_DIR / "bayes_factor.py"

TEST_FILE.chmod(0o444)

MODEL = "gemma-4-31b-it"
MAX_ATTEMPTS = 10
INCLUDE_TEST_FILE = False
PROMPT_FILE = TASK_DIR / "task.txt"

def run_tests() -> tuple[int, str]:
    result = subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", str(TEST_DIR)],
        cwd=TASK_DIR,
        capture_output=True,
        text=True,
    )
    return result.returncode, (result.stdout + result.stderr).strip()

client = GeminiSimpleAPI(
    api_key_file=Path("/workspace/secrets/gemini.json"),
    model=MODEL,
    working_dir=TASK_DIR,
    protected_directories=[TEST_DIR],
)

prompt_text = PROMPT_FILE.read_text()

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\n=== Attempt {attempt} ===")
    files, notes = client.prompt(
        prompt=prompt_text,
        attachments=[TEST_FILE] if INCLUDE_TEST_FILE else [],
        verbose=True,
    )

    code, output = run_tests()
    print(f"Output: {output}")
    (TASK_DIR / f"attempt_{attempt}").mkdir(parents=True, exist_ok=True)
    (TASK_DIR / f"attempt_{attempt}" / "output.txt").write_text(output)
    (TASK_DIR / f"attempt_{attempt}" / "prompt.txt").write_text(prompt_text)
    for file in files:
        shutil.copy(file, TASK_DIR / f"attempt_{attempt}" / file.name)
    
    if code == 0:
        print(f"\nTests passed on attempt {attempt}.")
        break
    prompt_text += (
        f"\n\n## Attempt {attempt} failed\n"
        f"```\n{output}\n```\n"
        "Fix the failures above."
    )
else:
    print(f"\nStopped after {MAX_ATTEMPTS} attempts; tests still failing.")
    sys.exit(1)