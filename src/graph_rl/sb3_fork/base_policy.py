"""Policies: abstract base class and concrete implementations."""

# Standard library
from abc import ABC
from typing import Any, Optional, Union

# Third-party
import torch as th
from gymnasium import spaces
from torch import nn

# First-party
from graph_rl.ppo.layers import BaseFeatureExtractor


class BasePolicy(nn.Module, ABC):
    """
    The base model object: makes predictions in response to observations.

    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    optimizer: th.optim.Optimizer
    features_extractor: BaseFeatureExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: type[BaseFeatureExtractor] = BaseFeatureExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return th.device("cpu")

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path: str, device: Union[th.device, str] = "auto"):
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        # Note(antonin): we cannot use `weights_only=True` here because we need to allow
        # gymnasium imports for the policy to be loaded successfully
        saved_variables = th.load(path, map_location=device, weights_only=False)

        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @staticmethod
    def _get_last_linear(module: nn.Module) -> nn.Linear:
        for m in reversed(list(module.modules())):
            if isinstance(m, nn.Linear):
                return m
        raise RuntimeError("Couldn't find a Linear layer inside the module to initialize.")

    def _init_head_output(self, head: nn.Module, *, weight_gain: float, bias_value: float) -> None:
        last_linear = self._get_last_linear(head)
        nn.init.orthogonal_(last_linear.weight, gain=weight_gain)
        nn.init.constant_(last_linear.bias, bias_value)
