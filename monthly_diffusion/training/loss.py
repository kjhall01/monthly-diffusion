
import torch 

def three_month_loss(model, batch, mode):
    atmosphere = batch["atmosphere"]
    ocean = batch["ocean"]
    months = batch["month"]
    co2 = batch["co2"]

    atmosphere, ocean, co2 = model.preprocess(atmosphere, ocean, co2)

    batch_size = atmosphere.shape[0]

    # First two months initialize both histories.
    atmos_previous = atmosphere[:, 0]
    atmos_current = atmosphere[:, 1]

    ocean_previous = ocean[:, 0]
    ocean_current = ocean[:, 1]

    # Keep CO2 availability fixed throughout this training trajectory.
    # The actual CO2 value can still change from month to month.
    drop_co2 = torch.zeros(
        batch_size, 1,
        dtype=torch.bool,
        device=atmosphere.device,
    )
    if model.training:
        drop_co2 = (
            torch.rand(batch_size, 1, device=atmosphere.device)
            < model.co2_dropout
        )

    previous_noise = (None, None)
    losses = []

    # Forecast sequence positions 2, 3, and 4.
    for target_index in range(2, 5):
        atmos_history = torch.cat(
            [atmos_current, atmos_previous], dim=-1,
        )
        ocean_history = torch.cat(
            [ocean_current, ocean_previous], dim=-1,
        )

        target_month = months[:, target_index]
        target_co2 = co2[:, target_index]

        # Denoising loss for this month's observed target.
        # History may now contain generated states.
        loss = model.loss(
            target_atmos=atmosphere[:, target_index],
            target_ocean=ocean[:, target_index],
            atmos_history=atmos_history,
            ocean_history=ocean_history,
            month=target_month,
            co2=target_co2,
            mode=mode,
            drop_co2=drop_co2,
        )
        losses.append(loss)

        # No need to generate the last month: no later history uses it.
        if target_index == 4:
            break

        predicted_atmos, predicted_ocean, previous_noise = model.sample(
            atmos_history=atmos_history,
            ocean_history=ocean_history,
            month=target_month,
            co2=target_co2,
            mode=mode,
            previous_noise=previous_noise,
            guidance_scale=1.0,
            drop_co2=drop_co2,
        )

        # Prescribed components advance using observed states.
        # Free components advance using generated states.
        if mode == "prescribed_atmosphere":
            next_atmos = atmosphere[:, target_index]
        else:
            next_atmos = predicted_atmos

        if mode == "prescribed_ocean":
            next_ocean = ocean[:, target_index]
        else:
            next_ocean = predicted_ocean

        atmos_previous, atmos_current = atmos_current, next_atmos
        ocean_previous, ocean_current = ocean_current, next_ocean

    # Equal weight for all three forecast months.
    return torch.stack(losses).mean()