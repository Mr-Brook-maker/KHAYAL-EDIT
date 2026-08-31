"""
Main Entry Point
CLI interface for the Image Editing Agent.
Supports single-shot and interactive conversation modes.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

from agent.orchestrator import ImageEditingAgent


def print_result(result: dict):
    print("\n" + "═" * 60)
    print("🤖 AGENT RESPONSE")
    print("═" * 60)
    print(result["output"])

    if result["steps"]:
        print(f"\n📋 Tool Calls ({len(result['steps'])} step(s)):")
        for i, step in enumerate(result["steps"], 1):
            status = "✅" if '"status": "success"' in step["output"] else "❌"
            print(f"  {i}. {status} {step['tool']}")
            try:
                obs = json.loads(step["output"])
                if obs.get("status") == "success":
                    print(f"     → {obs.get('output_path', '')}")
            except:
                pass

    if result.get("final_output_path"):
        print(f"\n📁 Final output: {result['final_output_path']}")

    print("═" * 60 + "\n")


def interactive_mode(agent: ImageEditingAgent):
    """REPL loop for multi-turn editing sessions."""
    print("\n🎨 Image Editing Agent — Interactive Mode")
    print("Type 'quit' to exit, 'reset' to clear memory\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset_memory()
            print("Memory cleared.\n")
            continue

        result = agent.run(user_input)
        print_result(result)


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal AI Image Editing Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -p "Remove background from photo.jpg and save as PNG"
  python main.py -i photo.jpg -p "Make it cinematic and resize to 1920x1080"
  python main.py --interactive
        """
    )
    parser.add_argument("-p", "--prompt", help="Editing prompt (single-shot mode)")
    parser.add_argument("-i", "--image", help="Input image path")
    parser.add_argument("--interactive", action="store_true", help="Start interactive session")

    args = parser.parse_args()

    if not args.prompt and not args.interactive:
        parser.print_help()
        sys.exit(1)

    agent = ImageEditingAgent()

    if args.interactive:
        interactive_mode(agent)
    else:
        result = agent.run(args.prompt, image_path=args.image)
        print_result(result)


if __name__ == "__main__":
    main()