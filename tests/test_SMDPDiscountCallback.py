"""Tests that SMDPDiscountCallback stores each decision's discount where the rollout buffer will add the decision."""

from types import SimpleNamespace

import numpy as np

from HUGIN_gym.callbacks.SMDPDiscountCallback import SMDPDiscountCallback


def test_discounts_are_written_at_the_buffer_position() -> None:
    # Arrange
    buffer = SimpleNamespace(pos=1, discounts=np.full((3, 2), 0.9, dtype=np.float32))
    callback = SMDPDiscountCallback()
    callback.model = SimpleNamespace(rollout_buffer=buffer)
    callback.update_locals({"infos": [{"discount": 0.81}, {"discount": 0.59}]})

    # Act
    keep_training = callback._on_step()

    # Assert
    assert keep_training
    np.testing.assert_allclose(buffer.discounts, [[0.9, 0.9], [0.81, 0.59], [0.9, 0.9]], rtol=1e-6)
