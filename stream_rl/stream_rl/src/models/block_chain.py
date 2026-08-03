"""Chains Block-protocol modules sequentially, exposing the same
(inputs, done, initial_carry) -> (carry, output) interface as a single Block."""
import flax.linen as nn


class BlockChain(nn.Module):
    blocks: list

    @nn.compact
    def __call__(self, inputs, done=None, initial_carry=None, **kwargs):
        x = inputs
        carries = initial_carry if initial_carry is not None else [None] * len(self.blocks)
        new_carries = []
        for block, carry in zip(self.blocks, carries):
            result = block(x, done=done, initial_carry=carry, **kwargs)
            match result:
                case (c, out):
                    pass
                case out:
                    c = None
            new_carries.append(c)
            x = out
        return tuple(new_carries), x

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return tuple(
            getattr(block, "initialize_carry", lambda k, s: None)(key, input_shape)
            for block in self.blocks
        )