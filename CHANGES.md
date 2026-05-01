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