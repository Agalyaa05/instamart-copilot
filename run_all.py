"""run_all.py - Complete end-to-end execution of Instamart Copilot pipeline.
Executes Part A, checkpoint validation, Part B, Part C, and Part D in order.
"""

import sys
import subprocess

def run_step(cmd_desc, args):
    print(f"\n=======================================================")
    print(f"STEP: {cmd_desc}")
    print(f"COMMAND: {' '.join(args)}")
    print(f"=======================================================")
    res = subprocess.run(args, capture_output=False)
    if res.returncode != 0:
        print(f"ERROR: Step '{cmd_desc}' failed with exit code {res.returncode}")
        sys.exit(res.returncode)

def main():
    py_exec = sys.executable
    run_step("Data Generation", [py_exec, "generate_data.py"])
    run_step("Part A - Data Foundation", [py_exec, "src/part_a.py"])
    run_step("Validate Checkpoints", [py_exec, "src/validate_checkpoints.py"])
    run_step("Part B - Customer Segments & Lapse Prediction", [py_exec, "src/part_b.py"])
    run_step("Part C - Support Ticket Triage with Guardrails", [py_exec, "src/part_c.py"])
    run_step("Part D - Policy RAG & Refund Agent", [py_exec, "src/part_d.py"])
    print("\n=======================================================")
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY")
    print("=======================================================")

if __name__ == "__main__":
    main()
