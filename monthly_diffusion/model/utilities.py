
from .encoder import  MeshStage, RefinedIcosahedralMeshEncoder
from .decoder import RefinedIcosahedralMeshDecoder
from .conditions import  ScalarToLatentChannels
import torch

def _load_models(checkpoint, preprocessor, config, device):
    condition_channels = config["condition_channels"]
    encoder = RefinedIcosahedralMeshEncoder(
        hierarchy=[MeshStage(**stage) for stage in config["encoder_stages"]],
        preprocessor=preprocessor, channels=preprocessor.channels + condition_channels,
        latent_channels=config["latent_channels"],
    )
    decoder = RefinedIcosahedralMeshDecoder(
        hierarchy=[MeshStage(**stage) for stage in config["decoder_stages"]],
        preprocessor=preprocessor, channels=preprocessor.channels,
        latent_channels=config["latent_channels"] + condition_channels,
    )
    ec = ScalarToLatentChannels(config["n_scalars"], encoder.grid, condition_channels)
    dc = ScalarToLatentChannels(config["n_scalars"], decoder.meshes[0], condition_channels)
    for key, model in zip(("encoder", "decoder", "ec", "dc"), (encoder, decoder, ec, dc)):
        # Encoder/decoder both contain the same fitted preprocessor buffers.
        for name, value in checkpoint["preprocessor"].items():
            nested = checkpoint[key].get(f"preprocessor.{name}")
            if nested is not None and not torch.equal(value, nested):
                raise ValueError(f"Inconsistent checkpoint preprocessing in {key}: {name}")
        model.load_state_dict(checkpoint[key], strict=True)
        model.to(device).eval().requires_grad_(False)
    return encoder, decoder, ec, dc