"""Test SMPO single-output edge case (apply with spacing >= L)."""

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
        output_mps = smpo.apply(input_mps)

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
        output_mps = smpo.apply(input_mps)

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
        output_mps = smpo.apply(input_mps)

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
        output_mps = smpo.apply(input_mps)

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
        output_mps = smpo.apply(input_mps)

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
        output_mps = smpo.apply(input_mps)

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
        output_mps = smpo.apply(input_mps)

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
        output_mps = smpo.apply(input_mps)

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
        output_mps = smpo.apply(input_mps)

        assert output_mps.L == 1, f"Expected 1-site output for L={L}, got {output_mps.L}"

class TestSMPOArbitraryOutputPositions:
    """Tests for SMPO with arbitrary output positions via output_inds parameter."""

    def test_single_output_center(self):
        """Single output at center position (output_inds=[9] for L=19)."""
        L = 19
        center = L // 2  # 9
        key = jax.random.PRNGKey(100)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=[center],
            bond_dim=8,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        assert smpo.leading_gap == center
        assert smpo.output_positions == [center]

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 1, f"Expected 1-site output, got {output_mps.L}"

    def test_single_output_last_position(self):
        """Single output at last position (output_inds=[L-1])."""
        L = 10
        key = jax.random.PRNGKey(101)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=[L - 1],
            bond_dim=4,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        assert smpo.leading_gap == L - 1
        assert smpo.output_positions == [L - 1]

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 1, f"Expected 1-site output, got {output_mps.L}"

    def test_single_output_first_position(self):
        """Single output at first position (output_inds=[0]) - same as spacing=L."""
        L = 10
        key = jax.random.PRNGKey(102)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=[0],
            bond_dim=4,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        assert smpo.leading_gap == 0
        assert smpo.output_positions == [0]

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 1, f"Expected 1-site output, got {output_mps.L}"

    def test_multiple_arbitrary_outputs(self):
        """Multiple outputs at arbitrary positions (output_inds=[3, 9, 15])."""
        L = 19
        output_positions = [3, 9, 15]
        key = jax.random.PRNGKey(103)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=output_positions,
            bond_dim=8,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        assert smpo.leading_gap == 3
        assert smpo.output_positions == output_positions

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 3, f"Expected 3-site output, got {output_mps.L}"

    def test_multiple_outputs_including_first(self):
        """Multiple outputs including position 0 (output_inds=[0, 5, 10])."""
        L = 15
        output_positions = [0, 5, 10]
        key = jax.random.PRNGKey(104)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=output_positions,
            bond_dim=4,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        assert smpo.leading_gap == 0
        assert smpo.output_positions == output_positions

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 3, f"Expected 3-site output, got {output_mps.L}"

    def test_multiple_outputs_including_last(self):
        """Multiple outputs including last position (output_inds=[0, 7, 14])."""
        L = 15
        output_positions = [0, 7, 14]
        key = jax.random.PRNGKey(105)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=output_positions,
            bond_dim=4,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        assert smpo.leading_gap == 0
        assert smpo.output_positions == output_positions

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 3, f"Expected 3-site output, got {output_mps.L}"

    def test_two_outputs_center_biased(self):
        """Two outputs biased toward center (output_inds=[4, 10] for L=15)."""
        L = 15
        output_positions = [4, 10]
        key = jax.random.PRNGKey(106)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=output_positions,
            bond_dim=4,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        assert smpo.leading_gap == 4
        assert smpo.output_positions == output_positions

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 2, f"Expected 2-site output, got {output_mps.L}"

    @pytest.mark.parametrize("center", [1, 2, 3, 4])
    def test_small_system_center_output(self, center):
        """Small systems with center output."""
        L = center * 2 + 1  # Ensure center is valid
        key = jax.random.PRNGKey(110 + center)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=[center],
            bond_dim=4,
            phys_dim=(2, 2),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_product_state_mps(L, phys_dim=2)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 1, f"Expected 1-site output for L={L}, center={center}, got {output_mps.L}"


class TestSMPOArbitraryOutputsWithRandomMPS:
    """Test arbitrary output positions with non-trivial input bond dimensions."""

    def test_center_output_random_mps(self):
        """Center output with random MPS input."""
        L = 19
        center = L // 2
        key = jax.random.PRNGKey(200)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=[center],
            bond_dim=8,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_random_mps(L, phys_dim=3, bond_dim=4)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 1, f"Expected 1-site output, got {output_mps.L}"

    def test_multiple_arbitrary_random_mps(self):
        """Multiple arbitrary outputs with random MPS input."""
        L = 19
        output_positions = [3, 9, 15]
        key = jax.random.PRNGKey(201)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=output_positions,
            bond_dim=8,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_random_mps(L, phys_dim=3, bond_dim=4)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 3, f"Expected 3-site output, got {output_mps.L}"


class TestSMPOOutputIndsValidation:
    """Test validation of output_inds parameter."""

    def test_invalid_index_negative(self):
        """Negative index should raise ValueError."""
        with pytest.raises(ValueError, match="invalid index"):
            SMPO_initialize(
                L=10,
                initializer=jax.nn.initializers.normal(stddev=0.1),
                key=jax.random.PRNGKey(300),
                output_inds=[-1, 5],
                bond_dim=4,
                phys_dim=(2, 2),
            )

    def test_invalid_index_too_large(self):
        """Index >= L should raise ValueError."""
        with pytest.raises(ValueError, match="invalid index"):
            SMPO_initialize(
                L=10,
                initializer=jax.nn.initializers.normal(stddev=0.1),
                key=jax.random.PRNGKey(301),
                output_inds=[0, 10],
                bond_dim=4,
                phys_dim=(2, 2),
            )

    def test_invalid_unsorted_indices(self):
        """Unsorted indices should raise ValueError."""
        with pytest.raises(ValueError, match="sorted"):
            SMPO_initialize(
                L=10,
                initializer=jax.nn.initializers.normal(stddev=0.1),
                key=jax.random.PRNGKey(302),
                output_inds=[5, 3, 8],
                bond_dim=4,
                phys_dim=(2, 2),
            )

    def test_invalid_duplicate_indices(self):
        """Duplicate indices should raise ValueError."""
        with pytest.raises(ValueError, match="sorted"):
            SMPO_initialize(
                L=10,
                initializer=jax.nn.initializers.normal(stddev=0.1),
                key=jax.random.PRNGKey(303),
                output_inds=[3, 3, 8],
                bond_dim=4,
                phys_dim=(2, 2),
            )


class TestSMPOOutputNormFinite:
    """Ensure output norms are finite for various configurations."""

    @pytest.mark.parametrize("output_pos", [[0], [4], [9], [3, 9, 15], [0, 9, 18]])
    def test_norm_finite_various_positions(self, output_pos):
        """Output norm should be finite for various output configurations."""
        L = 19
        key = jax.random.PRNGKey(400)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            output_inds=output_pos,
            bond_dim=8,
            phys_dim=(3, 3),
            cyclic=False,
            boundary='obc'
        )

        input_mps = make_product_state_mps(L, phys_dim=3)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        for i, t in enumerate(output_mps.tensors):
            norm_val = float(jnp.linalg.norm(t.data))
            assert np.isfinite(norm_val), f"Tensor {i} norm is not finite: {norm_val}"
            assert norm_val > 0, f"Tensor {i} norm is zero"


class TestSMPOBackwardCompatSpacingVsOutputInds:
    """Verify that spacing parameter and equivalent output_inds produce same structure."""

    def test_spacing_equals_output_inds_structure(self):
        """SMPO with spacing=5 should have same output_positions as output_inds=[0,5]."""
        L = 10
        key = jax.random.PRNGKey(500)

        smpo_spacing = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=5,
            bond_dim=4,
            phys_dim=(2, 2),
            cyclic=False,
            boundary='obc'
        )

        assert smpo_spacing.output_positions == [0, 5]
        assert smpo_spacing.leading_gap == 0

    def test_uniform_spacing_backward_compat(self):
        """Uniform spacing should still work and produce correct outputs."""
        L = 12
        key = jax.random.PRNGKey(501)
        smpo = SMPO_initialize(
            L=L,
            initializer=jax.nn.initializers.normal(stddev=0.1),
            key=key,
            spacing=4,
            bond_dim=4,
            phys_dim=(2, 2),
            cyclic=False,
            boundary='obc'
        )

        assert smpo.output_positions == [0, 4, 8]
        assert smpo.leading_gap == 0

        input_mps = make_product_state_mps(L, phys_dim=2)
        output_mps = smpo.apply(input_mps, normalize_on_contract=False)

        assert output_mps.L == 3, f"Expected 3-site output, got {output_mps.L}"