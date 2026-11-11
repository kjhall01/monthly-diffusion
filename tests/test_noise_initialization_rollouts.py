"""The initialization comparison must pair innovations and preserve history order."""

import numpy as np
import torch

from test_gaussian_forecast import ToyModel

from monthly_diffusion.experiments import compare_noise_initialization_rollouts as audit


def test_paired_noise_current_previous_and_exact_resume(tmp_path):
    initial = [torch.stack((torch.full((2, 32, 4), 2.), torch.ones(2, 32, 4)), dim=1)
               for _ in range(2)]
    args = dict(first="1967-01", members=2, seed=43, co2=340.)
    model = ToyModel()
    audit.paired_rollouts(model, initial, tmp_path, years=1, **args)
    histories = torch.load(tmp_path/"initial.pt", weights_only=False)
    first = torch.load(tmp_path/"year-001.pt", weights_only=False)
    for i, component in enumerate(audit.COMPONENTS):
        np.testing.assert_array_equal(histories["encoded_noise"]["history"][i][:, 0], 2.)
        np.testing.assert_array_equal(histories["encoded_noise"]["history"][i][:, 1], 1.)
        delta = histories["encoded_noise"]["history"][i]-histories["direct_latent"]["history"][i]
        actual = first["latents"]["encoded_noise"][component][0]-first["latents"]["direct_latent"][component][0]
        np.testing.assert_allclose(actual, (.8*delta[:, 0]+.1*delta[:, 1]).numpy(), atol=1.e-6)
    audit.paired_rollouts(model, initial, tmp_path, years=2, **args)
    uninterrupted = tmp_path/"uninterrupted"
    uninterrupted.mkdir()
    audit.paired_rollouts(ToyModel(), initial, uninterrupted, years=2, **args)
    resumed = torch.load(tmp_path/"year-002.pt", weights_only=False)
    direct = torch.load(uninterrupted/"year-002.pt", weights_only=False)
    for kind in audit.KINDS:
        for component in audit.COMPONENTS:
            np.testing.assert_array_equal(resumed["latents"][kind][component], direct["latents"][kind][component])
    assert model.decoded == 0


def test_member_metrics_do_not_splice_member_boundaries():
    t = np.arange(360)
    values = np.column_stack([np.sin(2*np.pi*t/48), np.cos(2*np.pi*t/48)])
    result = audit.member_metrics(values)
    np.testing.assert_allclose(result["std"], np.std(values, axis=0))
    np.testing.assert_allclose(result["acf"][:, 0], 1.)
    assert (result["acf"][:, 24] < -.8).all()
