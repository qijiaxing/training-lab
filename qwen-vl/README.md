# Qwen-VL

This is an example showing Qwen-VL model training optimization.

## Profiling

To use pytorch profiler:
```python
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=10),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    record_shapes=True,
    profile_memory=True,
    with_stack=False
) as prof:
```

After running, json files will be saved in `<trace_dir>`. To visualize:
```
# pip install torch_tb_profiler
tensorboard --logdir=<trace_dir>
```
