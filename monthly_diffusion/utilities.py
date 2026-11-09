import sys 
import torch 
import xarray as xr 
from .model import _load_models , Preprocessor

class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)

def close_diagnostics(console_stdout, console_stderr, diagnostics_file):
    sys.stdout, sys.stderr = console_stdout, console_stderr
    diagnostics_file.close()



def load_models(HOME_DIR, component):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    checkpoint = torch.load(
        HOME_DIR / component /  "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]
    coords = xr.Dataset(coords={
        **{
            name: values.numpy()
            for name, values in checkpoint["coordinates"].items()
        },
        "variable": config["variables"],
    }).coords

    preprocessor = Preprocessor.from_state_dict(
        checkpoint["preprocessor"], coords
    )

    encoder, decoder, ec, dc = _load_models(
        checkpoint, preprocessor, config, device
    )
    statics = checkpoint["statics"].to(device)

    # Models and preprocessor have already been constructed and restored.
    for module in (encoder, decoder, ec, dc, preprocessor):
        module.to(device).eval()

    return (encoder, decoder, ec, dc, preprocessor, config)
