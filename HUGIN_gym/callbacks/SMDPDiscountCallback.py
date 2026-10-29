from stable_baselines3.common.callbacks import BaseCallback


class SMDPDiscountCallback(BaseCallback):
    """Copies the discount of each meta decision (info["discount"] = gamma**option_length) into SMDPDictRolloutBuffer.

    SB3 calls the callbacks after each env step and before the step is added to the rollout buffer, so the
    buffer's current position is where this step will be stored.

    SB3 bootstraps truncated episodes with the model's fixed gamma, not the decision's own discount. With the HRL
    gamma of 0.998 per step this changes the bootstrap value by less than 1 %, once per episode.
    """

    def _on_step(self) -> bool:
        buffer = self.model.rollout_buffer
        buffer.discounts[buffer.pos] = [info["discount"] for info in self.locals["infos"]]
        return True
