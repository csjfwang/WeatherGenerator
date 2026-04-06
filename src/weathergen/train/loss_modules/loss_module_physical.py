# pylint: disable=bad-builtin
# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import re
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig

import weathergen.train.loss_modules.loss_functions as loss_fns
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.train.utils import TRAIN, VAL, Stage

_logger = logging.getLogger(__name__)


TAROT_LOSS_GROUP_PATTERNS = {
    "dynamic": [r"^10u$", r"^10v$", r"^msl$", r"^u_", r"^v_", r"^z_"],
    "thermodynamic": [r"^2d$", r"^2t$", r"^q_", r"^t_"],
    "pressure_50_100": [r"^[a-z0-9]+_50$", r"^[a-z0-9]+_100$"],
    "pressure_50_250": [
        r"^[a-z0-9]+_50$",
        r"^[a-z0-9]+_100$",
        r"^[a-z0-9]+_150$",
        r"^[a-z0-9]+_200$",
        r"^[a-z0-9]+_250$",
    ],
    "pressure_300_600": [r"^[a-z0-9]+_300$", r"^[a-z0-9]+_400$", r"^[a-z0-9]+_500$", r"^[a-z0-9]+_600$"],
    "pressure_700_1000_surface": [
        r"^[a-z0-9]+_700$",
        r"^[a-z0-9]+_850$",
        r"^[a-z0-9]+_925$",
        r"^[a-z0-9]+_1000$",
        r"^10u$",
        r"^10v$",
        r"^2d$",
        r"^2t$",
        r"^msl$",
        r"^sp$",
    ],
}


def get_num_samples(config) -> np.typing.NDArray:
    """
    Get number of samples in source/target config
    """
    return np.array([s_cfg.get("num_samples", 1) for _, s_cfg in config.items()])


class LossPhysical(LossModuleBase):
    """
    Manages and computes the overall loss for a WeatherGenerator model during
    training and validation stages.

    This class handles the initialization and application of various loss functions,
    applies channel-specific weights, constructs masks for missing data, and
    aggregates losses across different data streams, channels, and forecast steps.
    It provides both the main loss for backpropagation and detailed loss metrics for logging.
    """

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.mode_cfg = mode_cfg
        self.stage = stage
        self.device = device
        self.name = "LossPhysical"

        # dynamically load loss functions based on configuration and stage
        self.loss_fcts = [
            [
                getattr(loss_fns, name),
                params.get("weight", 1.0),
                name,
            ]
            for name, params in loss_fcts.items()
        ]
        self._tarot_channel_selection_cache: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, list[str]] | None] = {}

    def _get_tarot_channel_patterns(self) -> list[str]:
        if not (
            self.stage == VAL
            and bool(self.cf.get("tarot_export_features", False))
            and str(self.cf.get("tarot_feature_mode", "latent")).lower() == "gradient"
        ):
            return []

        explicit_patterns = [str(p).strip() for p in self.cf.get("tarot_loss_channel_patterns", []) if str(p).strip()]
        if explicit_patterns:
            return explicit_patterns

        group_name = str(self.cf.get("tarot_loss_channel_group", "")).strip().lower()
        if not group_name:
            return []
        if group_name not in TAROT_LOSS_GROUP_PATTERNS:
            valid = ", ".join(sorted(TAROT_LOSS_GROUP_PATTERNS))
            raise ValueError(
                f"Unsupported tarot_loss_channel_group='{group_name}'. Supported values: {valid}."
            )
        return TAROT_LOSS_GROUP_PATTERNS[group_name]

    def _select_tarot_validation_channels(
        self, stream_name: str, target_channels: list[str]
    ) -> tuple[np.ndarray, list[str]] | None:
        cache_key = (stream_name, tuple(target_channels))
        if cache_key in self._tarot_channel_selection_cache:
            return self._tarot_channel_selection_cache[cache_key]

        patterns = self._get_tarot_channel_patterns()
        if not patterns:
            self._tarot_channel_selection_cache[cache_key] = None
            return None

        regexes = [re.compile(pattern) for pattern in patterns]
        selected_idx = np.array(
            [i for i, channel in enumerate(target_channels) if any(regex.search(channel) for regex in regexes)],
            dtype=np.int64,
        )
        if selected_idx.size == 0:
            available = ", ".join(target_channels)
            raise ValueError(
                f"No target channels matched TAROT channel selection for stream '{stream_name}'. "
                f"Patterns={patterns}. Available channels: [{available}]"
            )

        selected_channels = [target_channels[int(i)] for i in selected_idx.tolist()]
        _logger.info(
            "TAROT validation loss channel selection for stream %s: %s",
            stream_name,
            selected_channels,
        )
        result = (selected_idx, selected_channels)
        self._tarot_channel_selection_cache[cache_key] = result
        return result

    def _get_weights(self, stream_info):
        """
        Get weights for current stream
        """

        device = self.device

        # Determine stream and channel loss weights based on the current stage
        if self.stage == TRAIN:
            # set loss_weights to 1. when not specified
            stream_info_loss_weight = stream_info.get("loss_weight", 1.0)
            weights_channels = (
                torch.tensor(stream_info["target_channel_weights"]).to(
                    device=device, non_blocking=True
                )
                if "target_channel_weights" in stream_info
                else None
            )
        elif self.stage == VAL:
            # in validation mode, always unweighted loss
            stream_info_loss_weight = 1.0
            weights_channels = None

        return stream_info_loss_weight, weights_channels

    def _get_output_step_weights(self, len_forecast_steps):
        timestep_weight_config = self.mode_cfg.get("forecast", {}).get("timestep_weight", {})
        if len(timestep_weight_config) == 0:
            return [1.0 for _ in range(len_forecast_steps)]
        weights_timestep_fct = getattr(loss_fns, list(timestep_weight_config.keys())[0])
        decay_factor = list(timestep_weight_config.values())[0]["decay_factor"]
        return weights_timestep_fct(len_forecast_steps, decay_factor)

    def _get_location_weights(self, stream_info, target_coords):
        location_weight_type = stream_info.get("location_weight", None)
        if location_weight_type is None:
            return None
        weights_locations_fct = getattr(loss_fns, location_weight_type)
        weights_locations = weights_locations_fct(target_coords)
        weights_locations = weights_locations.to(device=self.device, non_blocking=True)

        return weights_locations

    def _get_substep_masks(self, stream_info, output_step, target_times):
        """
        Find substeps and create corresponding masks (reused across loss functions)
        """

        tok_spacetime = stream_info.get("tokenize_spacetime", None)
        target_times_unique = np.unique(target_times) if tok_spacetime else [target_times]
        substep_masks = []
        for t in target_times_unique:
            # find substep
            mask_t = torch.tensor(t == target_times).to(self.device, non_blocking=True)
            substep_masks.append(mask_t)

        return substep_masks

    @staticmethod
    def _loss_per_loss_function(
        loss_fct,
        target: torch.Tensor,
        pred: torch.Tensor,
        substep_masks: list[torch.Tensor],
        weights_channels: torch.Tensor,
        weights_locations: torch.Tensor,
    ):
        """
        Compute loss for given loss function
        """

        loss_lfct = torch.tensor(0.0, device=target.device, requires_grad=True)
        losses_chs = torch.zeros(target.shape[-1], device=target.device, dtype=torch.float32)

        ctr_substeps = 0
        for mask_t in substep_masks:
            assert mask_t.sum() == len(weights_locations) if weights_locations is not None else True

            loss, loss_chs = loss_fct(
                target[mask_t], pred[:, mask_t], weights_channels, weights_locations
            )

            # accumulate loss
            loss_lfct = loss_lfct + loss
            losses_chs = losses_chs + loss_chs.detach() if len(loss_chs) > 0 else losses_chs
            ctr_substeps += 1 if loss > 0.0 else 0

        # normalize over forecast steps in window
        losses_chs /= ctr_substeps if ctr_substeps > 0 else 1.0

        # TODO: substep weight
        loss_lfct = loss_lfct / (ctr_substeps if ctr_substeps > 0 else 1.0)

        return loss_lfct, losses_chs

    def compute_loss(self, preds: dict, targets: dict, metadata) -> LossValues:
        """
        Computes the total loss for a given batch of predictions and corresponding
        stream data.

        The computed loss is:

        Mean_{stream}( Mean_{output_steps}( Mean_{loss_fcts}( loss_fct( target, pred, weigths) )))

        This method orchestrates the calculation of the overall loss by iterating through
        different data streams, forecast steps, channels, and configured loss functions.
        It applies weighting, handles NaN values through masking, and accumulates
        detailed loss metrics for logging.

        Args:
            preds: A nested list of prediction tensors. The outer list represents forecast steps,
                   the inner list represents streams. Each tensor contains predictions for that
                   step and stream.
            streams_data: A nested list representing the input batch data. The outer list is for
                          batch items, the inner list for streams. Each element provides an object
                          (e.g., dataclass instance) containing target data and metadata.

        Returns:
            A ModelLoss dataclass instance containing:
            - loss: The loss for back-propagation.
            - losses_all: A dictionary mapping stream names to a tensor of per-channel and
                          per-loss-function losses, normalized by non-empty targets/forecast steps.
            - stddev_all: A dictionary mapping stream names to a tensor of mean standard deviations
                          of predictions for channels with statistical loss functions, normalized.
        """

        # gradient loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        # counter for non-empty targets
        ctr_streams = 0

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all = defaultdict(dict)

        source2target_idxs, output_info, target2source_idxs, target_info = metadata

        # TODO: iterate over batch dimension
        for stream_info in self.cf.streams:
            stream_name = stream_info["name"]
            # TODO: avoid this
            target_channels = (
                stream_info.val_target_channels
                if self.stage == "val"
                else stream_info.train_target_channels
            )

            losses_all[stream_name] = defaultdict(dict)

            stream_loss_weight, weights_channels = self._get_weights(stream_info)

            # TODO: make nicer
            output_step_loss_weights = self._get_output_step_weights(len(targets.output_idxs))
            if len(targets.physical) - len(targets.output_idxs) > 0:
                output_step_loss_weights.insert(0, None)

            # loss_stream: loss for given stream
            loss_stream = torch.tensor(0.0, device=self.device, requires_grad=True)
            ctr_timesteps = 0
            for timestep_idx, (preds_cur, target_cur) in enumerate(
                zip(preds.physical, targets.physical, strict=True)
            ):
                preds_batch = preds_cur.get(stream_name, [])
                if not preds_batch:
                    # skip to next timestep if preds of current timestep are empty
                    continue

                targets_batch = target_cur[stream_name]["target"]
                targets_coords_batch = target_cur[stream_name]["target_coords"]
                targets_times_batch = target_cur[stream_name]["target_times"]
                targets_params = target_cur[stream_name]["target_metda_data"]
                targets_is_spoof = target_cur[stream_name]["is_spoof"]

                output_step_weight = output_step_loss_weights[timestep_idx]

                # loss_timestep: loss for given timestep
                loss_timestep = torch.tensor(0.0, device=self.device, requires_grad=True)
                ctr_batch = 0
                for pred, pred_params in zip(preds_batch, output_info, strict=True):
                    # source has a unique target but index is not invariant with multiple
                    # target_aux calculators
                    target_idx_native = pred_params.global_params.get("correspondence", -1)
                    target_idx = [
                        i
                        for i, t in enumerate(targets_params)
                        if t[stream_name].global_params["idx"] == target_idx_native
                    ]
                    # source/model_input has no target for physical loss
                    if len(target_idx) == 0:
                        continue
                    # source -> target correspondence has to be unique
                    assert len(target_idx) == 1
                    target_idx = target_idx[0]

                    # get weights for locations
                    weights_locations = self._get_location_weights(
                        stream_info, targets_coords_batch[target_idx]
                    )

                    # loss_st_corr: loss for give source-target correspondence
                    loss_st_corr = torch.tensor(0.0, device=self.device, requires_grad=True)
                    ctr_loss_fcts = 0
                    for loss_fct, loss_fct_weight, loss_fct_name in self.loss_fcts:
                        # skip is loss is not computed for this sample
                        if loss_fct_name not in pred_params.global_params["loss"]:
                            continue

                        target = targets_batch[target_idx]
                        target_times = targets_times_batch[target_idx]
                        pred_for_loss = pred
                        weights_channels_for_loss = weights_channels
                        selected_target_channels = target_channels

                        # spoofed inputs are masked in the output calculations
                        sw = 0.0 if targets_is_spoof[target_idx] else 1.0
                        spoof_weight = torch.tensor(sw, device=self.device, requires_grad=False)

                        # skip if either target or prediction has no data points
                        if not (target.shape[0] > 0 and pred.shape[0] > 0):
                            continue

                        # reshape prediction tensor to match target's dimensions: extract
                        # data/coords and remove token dimension if it exists.
                        # expected shape of pred is [ensemble_size, num_samples, num_channels].
                        pred = pred.reshape([pred.shape[0], *target.shape])
                        assert pred.shape[1] > 0

                        tarot_channel_selection = self._select_tarot_validation_channels(
                            stream_name, target_channels
                        )
                        if tarot_channel_selection is not None:
                            selected_idx, selected_target_channels = tarot_channel_selection
                            target = target[:, selected_idx]
                            pred_for_loss = pred[..., selected_idx]
                            if weights_channels is not None:
                                weights_channels_for_loss = weights_channels[selected_idx]
                        else:
                            pred_for_loss = pred

                        # get masks for sub-time steps
                        substep_masks = self._get_substep_masks(
                            stream_info, timestep_idx, target_times
                        )

                        losses_all[stream_name][str(timestep_idx)][loss_fct_name] = defaultdict(
                            dict
                        )
                        # loss_lfct: loss for given loss function aggregated over all channels
                        # loss_lfct_chs: loss for given loss function per channel
                        loss_lfct, loss_lfct_chs = self._loss_per_loss_function(
                            loss_fct,
                            target,
                            pred_for_loss,
                            substep_masks,
                            weights_channels_for_loss,
                            weights_locations,
                        )

                        for ch_n, v in zip(selected_target_channels, loss_lfct_chs, strict=True):
                            losses_all[stream_name][str(timestep_idx)][loss_fct_name][ch_n] = (
                                spoof_weight * v if v != 0.0 else torch.nan
                            )

                        # Add the weighted and normalized loss from this loss function to the total
                        # batch loss
                        loss_cur_w = spoof_weight * loss_fct_weight * loss_lfct * output_step_weight
                        loss_st_corr = loss_st_corr + loss_cur_w
                        ctr_loss_fcts += 1 if (loss_lfct > 0.0 and sw > 0.0) else 0

                    loss_timestep = loss_timestep + loss_st_corr
                    ctr_batch += 1 if ctr_loss_fcts > 0.0 else 0

                loss_stream = loss_stream + loss_timestep
                ctr_timesteps += 1 if ctr_batch > 0 else 0

            denom = ctr_timesteps if ctr_timesteps > 0 else 1.0
            loss = loss + (stream_loss_weight * loss_stream) / denom

            ctr_streams += 1 if ctr_timesteps > 0 else 0

        # normalize by all targets and forecast steps that were non-empty
        # (with each having an expected loss of 1 for an uninitalized neural net)
        if loss == 0.0:
            _logger.warning(
                "Loss is 0.0, likely incorrect configuration. Check stream"
                " support time and training configuration."
            )
        loss = loss / ctr_streams if ctr_streams > 0 else loss

        def _nested_dict():
            return defaultdict(dict)

        # Reorder losses_all to [stream_name][loss_fct_name][ch_n][output_step]
        reordered_losses = defaultdict(dict)
        for stream_name, output_step_dict in losses_all.items():
            reordered_losses[stream_name] = defaultdict(_nested_dict)
            for output_step, lfct_dict in output_step_dict.items():
                for loss_fct_name, ch_dict in lfct_dict.items():
                    for ch_n, v in ch_dict.items():
                        reordered_losses[stream_name][loss_fct_name][ch_n][output_step] = v

        # Calculate per stream, per lfct average across channels and output_steps
        for stream_name, lfct_dict in reordered_losses.items():
            for loss_fct_name, ch_dict in lfct_dict.items():
                reordered_losses[stream_name][loss_fct_name]["avg"] = 0
                count = 0
                for ch_n, output_step_dict in ch_dict.items():
                    if ch_n != "avg":
                        for _, v in output_step_dict.items():
                            reordered_losses[stream_name][loss_fct_name]["avg"] += v
                            count += 1
                reordered_losses[stream_name][loss_fct_name]["avg"] /= count

        # Return all computed loss components encapsulated in a ModelLoss dataclass
        return LossValues(loss=loss, losses_all=reordered_losses, stddev_all=None)
