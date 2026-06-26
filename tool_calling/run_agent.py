import argparse
import importlib.util
import sys
from pathlib import Path


def build_system_prompt(task: str, model_path: str, skill_names: list, tool_names: list) -> str:
    skills_list = "\n".join(f"  {s}" for s in skill_names)
    tools_list = ", ".join(["load_skill"] + tool_names)
    return f"""You are an interpretability researcher running a benchmark task.

Task: {task} probing
Target model to probe: {model_path}

Available skills — call load_skill("<name>") to get detailed instructions:
{skills_list}

Available tools:
  {tools_list}

All output files are written to the current working directory.
Do not ask for user input. Operate autonomously."""


def load_task_module(task: str, repo_root: Path):
    tools_path = repo_root / "src" / "tool_calling" / "tasks" / task / "tools.py"
    if not tools_path.exists():
        raise FileNotFoundError(f"No tools.py for task '{task}' at {tools_path}")
    spec = importlib.util.spec_from_file_location(f"task_tools_{task}", tools_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True, help="Orchestrator LLM")
    parser.add_argument("--model-path", required=True, help="HuggingFace model to probe")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root / "src" / "tool_calling"))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from tool_loop import run_tool_loop, set_skills_dir

    skills_dir = repo_root / "src" / "tool_calling" / "tasks" / args.task / "skills"
    set_skills_dir(skills_dir)
    skill_names = sorted(p.stem for p in skills_dir.glob("*.md")) if skills_dir.exists() else []

    task_module = load_task_module(args.task, repo_root)
    if hasattr(task_module, "set_output_dir"):
        task_module.set_output_dir(output_dir)
    tools = getattr(task_module, "TOOLS", {})

    system_prompt = build_system_prompt(
        task=args.task,
        model_path=args.model_path,
        skill_names=skill_names,
        tool_names=list(tools.keys()),
    )
    user_prompt = (
        f"Run the full {args.task} probing pipeline on {args.model_path}. "
        "Call load_skill to get instructions for each step before executing it."
    )

    result = run_tool_loop(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools,
        model=args.model,
    )
    print(result)


if __name__ == "__main__":
    main()
