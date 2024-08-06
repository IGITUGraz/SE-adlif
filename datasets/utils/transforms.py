import numpy as np
from dataclasses import dataclass
from typing import Union, Optional
import torch

@dataclass(frozen=True)
class Flatten:
    def __call__(self, data: np.ndarray):
        return data.reshape(data.shape[0], -1)

@dataclass(frozen=True)
class TakeEventByTime:
    """Take events in a certain time interval with a length proportional to
    a specified ratio of the original length.

    Parameters:
        duration_interval (Union[float, Tuple[float]], optional):
        the length of the taken time interval, expressed in a ratio of the original sequence duration.
            - If a float, the value is used to calculate the interval length (0, duration_ratio)
            - If a tuple of 2 floats, the taken interval is[min_val, max_val]
            Defaults to 0.2.

    Example:
        >>> transform = tonic.transforms.TakeEventByTime(duration_ratio=(0.1, 0.8))  # noqa
        (take the event part between 10% and 80%)
    """

    duration_interval: Union[float, tuple[float]] = 0.2

    def __call__(self, events):
        assert "x" and "t" and "p" in events.dtype.names
        # assert (
        #     type(self.duration_interval) == float and self.duration_interval >= 0.0 and self.duration_interval < 1.0
        # ) or (
        #     type(self.duration_interval) == tuple
        #     and len(self.duration_interval) == 2
        #     and all(val >= 0 and val < 1.0 for val in self.duration_interval)
        # )
        t_start = events["t"].min()

        t_end = events["t"].max()
        total_duration = t_end - t_start
        if isinstance(self.duration_interval, tuple):
            t_start = total_duration * self.duration_interval[0]
            t_end = total_duration * self.duration_interval[1]
        else:
            t_start = events["t"].min()
            t_end = total_duration * self.duration_interval
        mask_events = (events["t"] >= t_start) & (events["t"] <= t_end)
        mask_events = np.logical_not(mask_events)
        return np.delete(events, mask_events)


@dataclass(frozen=True)
class AddNoiseByproportion:
    """
    Add noise to a sample (in frame or event form)
    by proportion of available event space (H*W*p*T)

    If the sample is a frame that represent event count,
    we estimate the time-windows used for the frame transformation
    or we use the one given as parameters
    The available space is then H*W*p*T*time_windows

    The number of events to add is determined as
    floor(noise_prop*space)
    The events are then added uniformly to the sample.

    sensor_size must be defined if the sample is in events format.

    If the data has been binarized, you need to set time_window to 1
    or let it empty, the function estimate the maximum value in the sample
    which should be 1.
    """

    noise_prop: float
    is_frame: bool = False
    time_window: Optional[int] = None
    sensor_size: Optional[tuple] = None

    def __post_init__(self):
        if self.is_frame:
            call = self.frame_format_call
        else:
            if self.sensor_size is None:
                raise AttributeError("Sensor size cannot be None in Event mode")
            call = self.event_format_call
        setattr(AddNoiseByproportion, "__call__", call)

    def event_format_call(self, events):
        space = np.prod(self.sensor_size) * (events["t"].max() - events["t"].min())
        nb_spikes = int(np.floor(self.noise_prop * space))
        events = add_unique_triplets(events, self.sensor_size, nb_spikes)
        return events

    def frame_format_call(self, events):
        time_window = self.time_window
        if time_window is None:
            time_window = events.max()

        e_shape = events.shape
        e_flat = events.flatten()
        sum_e_flat = torch.sum(e_flat)
        # this is the empty "space" where new events may be added
        space = np.prod(e_shape) * time_window - sum_e_flat
        nb_spike = int(np.floor(space * self.noise_prop))

        while nb_spike != 0:
            # generate random events position uniformly on the space
            rand_index = np.random.randint(low=0, high=len(e_flat), size=(nb_spike,))
            # two position may be the same (for instance two events on the same time_window)
            # we count unique position
            rand_index, counts = np.unique(rand_index, return_counts=True)
            e_flat[rand_index] += torch.from_numpy(counts).float()
            # if we overshoot the available space for some position
            # nb_events > time_windows
            # we recompute new events at new positions
            e_flat_tmp = e_flat - time_window
            nb_spike = int(torch.sum(e_flat_tmp[e_flat_tmp >= 0]).item())
            e_flat = torch.clamp(e_flat, 0, time_window)
        events = torch.reshape(e_flat, e_shape)
        return events


# Function to generate random triplets
def add_unique_triplets(events, sensor_size, num_triplets):
    # for channel in events.dtype.names:
    #     if channel == "x":
    #         low, high = 0, sensor_size[0]
    #     if channel == "y":
    #         low, high = 0, sensor_size[1]
    #     if channel == "p":
    #         low, high = 0, sensor_size[2]
    #     if channel == "t":
    #         low, high = events["t"].min(), events["t"].max()
    max_x = sensor_size[0]
    max_t = events["t"].max()

    # Convert existing array to a set of tuples for fast lookup
    existing_set = set(map(tuple, events.tolist()))

    new_triplets = []
    while len(new_triplets) < num_triplets:
        # Generate a random quadruplet
        new_triplet = (np.random.randint(0, max_t), np.random.randint(0, max_x), 1)

        # Check if it is unique
        if new_triplet not in existing_set:
            new_triplets.append(new_triplet)
            existing_set.add(new_triplet)

    new_events = np.concatenate(
        [events, np.array(new_triplets, dtype=events.dtype)], axis=0
    )
    return np.sort(new_events, order="t")
