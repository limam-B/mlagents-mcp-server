# mlagents-mcp

MCP server for controlling Unity ML-Agents training runs from Claude Code.

Launch, stop, resume, monitor, compare, and export ML-Agents training — all through natural conversation without leaving your editor.

## Features

- **Training control** — start, stop, resume runs as background processes
- **Blocking waits** — tools that block until a condition is met (no sleep/poll loops needed)
- **Live monitoring** — read TensorBoard metrics, reward curves, and training logs in real time
- **Run comparison** — compare metrics across runs for hyperparameter tuning
- **Config management** — read and deep-merge update YAML training configs
- **Model export** — locate .onnx models and checkpoints
- **Two training modes** — Unity Editor (interactive) and built executable (headless batch)

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Unity ML-Agents `mlagents-learn` available in PATH (or via conda env)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI

## Installation

```bash
git clone https://github.com/limam-B/mlagents-mcp-server.git
cd mlagents-mcp-server
uv sync
```

## Quick setup with Claude Code

```bash
# Register the MCP server:
claude mcp add --scope project --transport stdio mlagents-training \
  -- uv run --directory /path/to/mlagents-mcp-server mlagents-mcp

# Unregister (from all scopes to clean up stale configs):
claude mcp remove --scope local mlagents-training
claude mcp remove --scope user mlagents-training
claude mcp remove --scope project mlagents-training

# List registered servers:
claude mcp list
```

### With environment variables

The server reads its configuration from environment variables. Add them to your `.mcp.json` (project-level) or pass them via the CLI:

```json
{
  "mcpServers": {
    "mlagents-training": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mlagents-mcp-server", "mlagents-mcp"],
      "env": {
        "MLAGENTS_PROJECT_ROOT": "/path/to/your/unity/project",
        "MLAGENTS_RESULTS_DIR": "results",
        "MLAGENTS_CONFIG_DIR": "config"
      }
    }
  }
}
```

### With conda (if ML-Agents is installed in a conda env)

```json
{
  "mcpServers": {
    "mlagents-training": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mlagents-mcp-server", "mlagents-mcp"],
      "env": {
        "MLAGENTS_PROJECT_ROOT": "/path/to/your/unity/project",
        "MLAGENTS_RESULTS_DIR": "results",
        "MLAGENTS_CONFIG_DIR": "config",
        "MLAGENTS_CONDA_ENV": "mlagents",
        "MLAGENTS_CONDA_PATH": "/home/user/miniconda3"
      }
    }
  }
}
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLAGENTS_PROJECT_ROOT` | `.` | Root directory of your Unity project |
| `MLAGENTS_RESULTS_DIR` | `results` | Training results directory (relative to project root) |
| `MLAGENTS_CONFIG_DIR` | `config` | Training config YAML directory (relative to project root) |
| `MLAGENTS_CONDA_ENV` | — | Conda environment name to activate before running `mlagents-learn` |
| `MLAGENTS_CONDA_PATH` | — | Path to conda installation (e.g. `/home/user/miniconda3`) |

## Tools (16)

### Training control

| Tool | Description |
|------|-------------|
| `force_training` | Launch a new training run (overwrites previous results). Blocks until ready by default. |
| `stop_training` | Gracefully stop a run (SIGINT, saves the model). |
| `resume_training` | Resume from checkpoint. Auto-reads config from previous run. |

### Monitoring

| Tool | Description |
|------|-------------|
| `get_run_status` | Status overview: reward trend, checkpoints, step progress. |
| `get_metrics` | Read TensorBoard scalars (reward, losses, learning rate, etc.). |
| `get_training_logs` | Tail live stdout/stderr from an active run. |
| `list_runs` | List all known runs with status filtering. |

### Comparison & export

| Tool | Description |
|------|-------------|
| `compare_runs` | Compare a metric across multiple runs (min/max/final + trend). |
| `export_model` | Locate .onnx model files and checkpoints. |

### Configuration

| Tool | Description |
|------|-------------|
| `get_config` | Read a YAML training config. |
| `update_config` | Deep-merge updates into a config (only specified keys change). |

### Blocking waits

These tools block until a condition is met, eliminating the need for sleep/poll loops in your AI workflows:

| Tool | Blocks until... |
|------|-----------------|
| `wait_for_first_metrics` | First TensorBoard data point appears. |
| `wait_for_step` | Training reaches a target step count. |
| `wait_for_reward` | Mean reward hits a threshold. |
| `wait_for_completion` | Run finishes (completed/failed/stopped). |
| `wait_for_checkpoint` | A new .onnx file appears on disk. |

## Two training modes

### Editor mode (no `env_path`)

Training connects to the Unity Editor. `force_training` blocks until `mlagents-learn` prints "Listening on port... press Play", then you (or an AI agent) presses Play in Unity.

```
force_training(config_path="movement.yaml", run_id="Movement_v1")
# → blocks until "Listening on port 5004. Start training by pressing Play..."
```

### Batch mode (with `env_path`)

Training launches a built executable directly — no Unity Editor needed. `force_training` blocks until the executable connects.

```
force_training(
    config_path="movement.yaml",
    run_id="Movement_v1",
    env_path="/path/to/Build.x86_64",
    num_envs=12,
    no_graphics=True,
)
# → blocks until "Connected to Unity environment"
```

## Example workflow

A typical automated training session:

```
1. force_training(config, run_id, ...)     # launch, blocks until ready
2. wait_for_first_metrics(run_id)          # blocks until data flowing
3. wait_for_step(run_id, 500000)           # blocks until milestone
4. get_metrics(run_id)                     # check reward curve
5. wait_for_completion(run_id)             # blocks until done
6. export_model(run_id)                    # find the .onnx file
```

No `bash sleep`. No retry loops. Each call blocks and returns when ready.

## Development

```bash
# Install dev dependencies:
uv sync --group dev

# Lint:
uv run ruff check src/

# Format:
uv run ruff format src/

# Run the server directly (stdio):
uv run mlagents-mcp
```

## Project structure

```
src/mlagents_mcp/
  server.py            # FastMCP app, all 16 tool definitions, entry point
  process_manager.py   # Subprocess launch/stop, log capture, port assignment
  metrics_reader.py    # TensorBoard event file parsing
  config_manager.py    # YAML config read/write/deep-merge
  run_registry.py      # Thread-safe run tracking + historical disk scan
  waiters.py           # Blocking wait logic for all wait_for_* tools
  types.py             # Shared dataclasses and enums
```

## License

MIT
