# Historical CLI workflows

These are the original `train_model.py`, `forecast.py` and `diagnose.py` CLI
workflows. The paired trainer now contains its own training loops; there is no
package training API. It retains combined AE validation, the original checkpoint
layout and the existing recipe flags, including resume.

New experiments use the [editable component scripts](../README.md) and their
shared run directory. Existing combined checkpoints remain loadable for inference.
