"""Module for testing immunity to probes with RL."""

# Load secrets from a repo-root .env (OPENAI_API_KEY, WANDB_API_KEY, HF_TOKEN, ...)
# so scripts and tests pick them up automatically. override=False (the default)
# means real/exported env vars win over .env, e.g. RunPod secrets.
try:
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    # python-dotenv not installed; set env vars another way.
    pass
