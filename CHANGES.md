1. Gated ``checkpoint_volume.commit()`` with ``if config.experiment.infra == "modal":`` to prevent crashes with Modal API key not being present, as the volume seems to want to commit to modal. (run_benchmark.py)

2. Replaced

```Gated
fig.write_image(
    trajectories_dir / f"phase_diagram_episode_{ep:03d}.png"
)
```

and

```
fig_gt.write_image(summary_dir / "phase_diagram_gt.png")
```

with a try-except, ie:

```python
try:
    fig.write_image(
        trajectories_dir / f"phase_diagram_episode_{ep:03d}.png"
    )
except Exception as exc:
    logger.warning(
        "Skipping phase diagram image export: %s",
        exc,
    )
```

(run_multi_systems.py)

which will prevent a crash involving Chrome not working correctly/not being present.

Warning: Due to strange kaleidoscope errors (requires 0.10 to fix a 0 tab problem, but this conflicts with the plotly version and is depreciated), I cannot get this working on Windows.

3. Confirmed the default chemeleon downloader is completely broken and will not download the file from the figshare link (but the figshare link works). I have downloaded it manually and uploaded to huggingface: https://huggingface.co/G-reen/chemeleon_reupload.

Created a script fix_chemeleon.py which clears out broken downloads, downloads the model from HF, and attempts to load it to check it it works. Running this script should fix the problems with the other scripts.

Also, you need ``weights_only=False`` for:

```python
self.dm = DiffusionModule.load_from_checkpoint(
    model_path, map_location=self.device,
)
```

4. As it turns out, it may have been running on CPU because torch did not install with CUDA due to ``uv sync`` defaulting to CPU-only torch installation. I have changed to pyproject.toml and updated the uv lock accordingly. Also pinned to a non-compromised version of pytorch lightning (see https://github.com/Lightning-AI/pytorch-lightning/security/advisories/GHSA-w37p-236h-pfx3)

Note: In addition, some other slight modifications were made to the pyproject.toml, to make corrections after some other packages changed versions.

5. Extremely substantial changes were made in ase_potential.py to prevent race conditions when loading and running parallelized MACE models.

6. Updated llm.py (both), llm_react_orchestrator.py to take sampling parameters so that it would be appropriate for the qwen models.

7. Updated llm.py in planners so that it would not erroneously crash when the llm outputs a misformatted message:
```python
try:
...
except Exception as e:
logger.error(f"LLMPlanner failed to generate or parse response: {e}")
class MockPred:
    compositions = []
pred = MockPred()
```

8. Testing langfuse integration for detailed logging on the llm prompts. Modified src/made/agents/__init__.py to initialize langfuse when a key is detected.