"""Test SMPO single-output edge case (apply_mps with spacing >= L)."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import quimb.tensor as qtn

from tn4ml.models.smpo import SMPO_initialize, SpacedMatrixProductOperator

jax.config.update("jax_enable_x64", False)


def make_product_state_mps(L, phys_dim=3, key=None):
    """Create a product-state MPS (bond dim 1) with L sites."""
    if key is None:
        key = jax.random.PRNGKey(123)
    arrays = []
    for i in range(L):
        key, subkey = jax.random.split(key)
        vec = jax.random.normal(subkey, shape=(phys_dim,))
        vec = vec / jnp.linalg.norm(vec)
        if i == 0:
            arrays.append(np.array(vec.reshape(1, phys_dim)))  # (r, p)
        elif i == L - 1:
            arrays.append(np.array(vec.reshape(1, phys_dim)))  # (l, p)
        else:
            arrays.append(np.array(vec.reshape(1, 1, phys_dim)))  # (l, r, p)
    return qtn.MatrixProductState(arrays, shape='lrp')


def make_random_mps(L, phys_dim=3, bond_dim=4, key=None):
    """Create a random MPS with specified bond dimension."""
    if key is None:
        key = jax.random.PRNGKey(456)
    arrays = []
    for i in range(L):
        key, subkey = jax.random.split(key)
        if i == 0:
            arr = jax.random.normal(subkey, shape=(bond_dim, phys_dim))
        elif i == L - 1:
            arr = jax.random.normal(subkey, shape=(bond_dim, phys_dim))
        else:
            arr = jax.random.normal(subkey, shape=(bond_dim, bond_dim, phys_dim))
        arrays.append(np.array(arr))
    return qtn.MatrixProductState(arrays, shape='lrp')


class TestSMPOSingleOutput:
    """Tests for SMPO with a single output site (19->1 compression)."""

    def test_single_output_product_state(self):
        """19->1 SMPO applied to a product-state MPS."""
        L = 19
        key = jax.random.PRNGKey(0)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=19,
            bond_dim=8,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply_mps(smpo, input_mps)

        assert output_mps.L == 1, f"Expected 1-site output, got {output_mps.L}"
        # The single tensor should have just the physical dimension
        output_shape = output_mps.tensors[0].shape
        assert len(output_shape) <= 2, (
            f"Single-site output tensor should be 1D or 2D, got shape {output_shape}"
        )

    def test_single_output_random_mps(self):
        """19->1 SMPO applied to a random MPS with bond dim > 1."""
        L = 19
        key = jax.random.PRNGKey(1)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=19,
            bond_dim=8,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_random_mps(L, phys_dim=3, bond_dim=4)
        output_mps = smpo.apply_mps(smpo, input_mps)

        assert output_mps.L == 1, f"Expected 1-site output, got {output_mps.L}"
        output_shape = output_mps.tensors[0].shape
        assert len(output_shape) <= 2, (
            f"Single-site output tensor should be 1D or 2D, got shape {output_shape}"
        )

    def test_single_output_small(self):
        """Minimal case: 3->1 SMPO."""
        L = 3
        key = jax.random.PRNGKey(2)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=3,
            bond_dim=4,
            phys_dim=(2, 2),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_product_state_mps(L, phys_dim=2)
        output_mps = smpo.apply_mps(smpo, input_mps)

        assert output_mps.L == 1, f"Expected 1-site output, got {output_mps.L}"

    def test_single_output_no_normalize(self):
        """19->1 without normalization during contraction."""
        L = 19
        key = jax.random.PRNGKey(3)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=19,
            bond_dim=8,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply_mps(smpo, input_mps, normalize_on_contract=False)

        assert output_mps.L == 1, f"Expected 1-site output, got {output_mps.L}"

    def test_single_output_norm_is_finite(self):
        """Output MPS norm should be finite and nonzero."""
        L = 19
        key = jax.random.PRNGKey(4)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=19,
            bond_dim=8,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply_mps(smpo, input_mps)

        norm_val = float(jnp.linalg.norm(output_mps.tensors[0].data))
        assert np.isfinite(norm_val), f"Output norm is not finite: {norm_val}"
        assert norm_val > 0, f"Output norm is zero"


class TestSMPOMultiOutputBackwardCompat:
    """Ensure multi-output cases still work after the fix."""

    def test_multi_output_spacing5(self):
        """Standard case: L=10, spacing=5 (outputs at 0, 5)."""
        L = 10
        key = jax.random.PRNGKey(10)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=5,
            bond_dim=4,
            phys_dim=(2, 2),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_product_state_mps(L, phys_dim=2)
        output_mps = smpo.apply_mps(smpo, input_mps)

        assert output_mps.L == 2, f"Expected 2-site output, got {output_mps.L}"

    def test_multi_output_spacing3(self):
        """L=10, spacing=3 (outputs at 0, 3, 6, 9)."""
        L = 10
        key = jax.random.PRNGKey(11)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=3,
            bond_dim=4,
            phys_dim=(2, 2),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_product_state_mps(L, phys_dim=2)
        output_mps = smpo.apply_mps(smpo, input_mps)

        # Outputs at 0, 3, 6, 9 -> 4 output sites
        # (Actually depends on how quimb handles L=10, spacing=3)
        assert output_mps.L >= 2, f"Expected multi-site output, got {output_mps.L}"

    def test_multi_output_with_random_mps(self):
        """Multi-output with non-trivial input bond dimension."""
        L = 10
        key = jax.random.PRNGKey(12)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=5,
            bond_dim=4,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_random_mps(L, phys_dim=3, bond_dim=4)
        output_mps = smpo.apply_mps(smpo, input_mps)

        assert output_mps.L == 2, f"Expected 2-site output, got {output_mps.L}"


class TestSMPOSingleOutputLargerBond:
    """Test single-output with various bond dimensions."""

    @pytest.mark.parametrize("bond_dim", [2, 4, 8, 16])
    def test_varying_bond_dim(self, bond_dim):
        """Single output with different SMPO bond dimensions."""
        L = 7
        key = jax.random.PRNGKey(20)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=L,
            bond_dim=bond_dim,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply_mps(smpo, input_mps)

        assert output_mps.L == 1, f"Expected 1-site output, got {output_mps.L}"

    @pytest.mark.parametrize("L", [3, 5, 10, 19])
    def test_varying_system_size(self, L):
        """Single output with different system sizes."""
        key = jax.random.PRNGKey(30)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=L,
            bond_dim=4,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply_mps(smpo, input_mps)

        assert output_mps.L == 1, f"Expected 1-site output for L={L}, got {output_mps.L}"
