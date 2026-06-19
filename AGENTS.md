# AGENTS.md

## Project context

This is a student collaborative project. The main branch should always stay runnable.

## Review guidelines

- Flag changes that may break the main training, inference, or evaluation pipeline.
- Flag hard-coded local absolute paths, such as /Users/..., /home/..., or C:\Users\....
- Flag private tokens, API keys, passwords, or credentials.
- Flag missing setup instructions when a new script, dependency, dataset, or checkpoint is required.
- Flag missing tests or at least missing smoke-test commands for important features.
- For ML code, check tensor shapes, device placement, dtype issues, batch dimensions, and train/eval mode mistakes.
- Prefer high-signal comments. Do not comment on tiny formatting issues unless they may cause bugs.
