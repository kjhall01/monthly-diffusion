from pathlib import Path 
import torch 
import xarray as xr
import pandas as pd 

from monthly_diffusion.model import ( MeshStage, RefinedIcosahedralMeshEncoder, 
    Preprocessor, RefinedIcosahedralMeshDecoder, 
    ScalarToLatentChannels, load_models
)

#HOME_DIR = Path("../output/" + RandomWords().get_random_word())
HOME_DIR = Path("../output/heptagon")
RUN_DIR = HOME_DIR / "atmosphere"
OUTPUT_DIR = RUN_DIR / "ae_trajectory/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA = Path("/Users/kyle/Dev/downloads/e3smv2_monthly_1deg_r1i1p1f1_185001-210012.nc")
BATCH_SIZE = 24

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

checkpoint = torch.load(
    RUN_DIR / "best.pt",
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

encoder, decoder, ec, dc = load_models(
    checkpoint, preprocessor, config, device
)
statics = checkpoint["statics"].to(device)

print(
    f"Restored epoch {checkpoint['epoch']} "
    f"(val MSE = {checkpoint['val_loss']:.6f})"
)

# Models and preprocessor have already been constructed and restored.
for module in (encoder, decoder, ec, dc, preprocessor):
    module.to(device).eval()


with xr.open_dataset(DATA) as ds:
    times = ds.time.values.copy()

    with torch.no_grad():
        for start in range(0, ds.sizes["time"], BATCH_SIZE):
            print(pd.Timestamp(ds.time.isel(time=start).values) )
            batch = ds.isel(time=slice(start, start + BATCH_SIZE))
            batch_da = batch[config["variables"]].to_array("variable")

            # Coordinate ordering is checked by xr_to_torch().
            state = preprocessor.xr_to_torch(batch_da, device=device)
            state = preprocessor.preprocess_torch(state)
            # New missing cells outside the training period must not propagate
            # through the network. Zero uses the saved training mean/background.
            state = torch.where(torch.isfinite(state), state, 0.0)

            months = torch.tensor(
                batch.time.dt.month.values,
                dtype=torch.float32,
                device=device,
            ).reshape(-1, 1)

            conditions = torch.cat([
                torch.cos(months / 12 * 2 * torch.pi),
                torch.sin(months / 12 * 2 * torch.pi),
            ], dim=-1)

            latent = encoder(
                torch.cat([state, ec(conditions)], dim=-1),
                statics,
            )

            reconstruction = decoder(
                torch.cat([latent, dc(conditions)], dim=-1),
                statics,
            )

            # Restore units, background, and invalid-location NaNs.
            if not torch.isfinite(reconstruction).all():
                raise ValueError(f"Non-finite reconstruction in batch starting at {start}; not saving")

            physical = preprocessor.postprocess_torch(reconstruction)
            
            # Inside the batch loop, after computing physical:
            recon_da = preprocessor.torch_to_xr(
                physical.cpu(),
                times=batch.time.values,
            )
            recon_ds = recon_da.to_dataset(dim="variable")

            recon_ds.to_netcdf(
                OUTPUT_DIR / f"reconstruction_{start:05d}.nc"
            )

            latent_da = xr.DataArray(
                data = latent.cpu(), 
                name='latent',
                dims = ('time', 'node', 'channel'), 
                coords = {
                    'time': batch.time.values,
                    'channel': range(latent.shape[2]),
                    'node' : range(latent.shape[1]),
                    'lat': ('node', encoder.meshes[-1].lat),
                    'lon': ('node', encoder.meshes[-1].lon),

                }
            )
            latent_da.to_netcdf(
                OUTPUT_DIR / f"latent_{start:05d}.nc"
            )

print('... success')
