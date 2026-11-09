from torch.utils.data import Dataset


class FiveMonthDataset(Dataset):
    """One continuous monthly trajectory, entirely within one data split."""

    def __init__(self, atmosphere, ocean, month, co2):
        # atmosphere: [T, N_atmos, C_atmos]
        # ocean:      [T, N_ocean, C_ocean]
        # month:      [T]
        # co2:        [T, 1]
        length = len(atmosphere)

        if length < 5:
            raise ValueError("Need at least five months")
        if not all(len(x) == length for x in (ocean, month, co2)):
            raise ValueError("All inputs must have matching time dimensions")

        self.atmosphere = atmosphere
        self.ocean = ocean
        self.month = month
        self.co2 = co2

    def __len__(self):
        return len(self.atmosphere) - 4

    def __getitem__(self, start):
        window = slice(start, start + 5)
        return {
            "atmosphere": self.atmosphere[window],
            "ocean": self.ocean[window],
            "month": self.month[window],
            "co2": self.co2[window],
        }