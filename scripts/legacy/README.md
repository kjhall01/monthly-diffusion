# Original paired workflow

The former `examples/train_model.py`, `examples/forecast.py`, and
`examples/diagnose.py` live here with their original CLI interfaces. The trainer
selects the two AEs together using combined validation loss, then trains diffusion.
Use it to resume an unfinished paired run.

New separate-stage runs follow the [main workflow](../README.md). Both paths use
the same model/checkpoint format. Finish partial separate-AE runs using the new
component scripts before switching to a convenience training API.
