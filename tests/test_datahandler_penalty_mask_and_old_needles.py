def _make_minimal_datahandler(torch, tmp_run_dir):
    from src.utils.datahandler import DataHandler

    dh = DataHandler(
        directory=str(tmp_run_dir),
        device="cuda",
        dtype=torch.float64,
        d=3,
    )
    # Minimal init data
    device = dh.device
    dtype = dh.dtype
    X_init_actual = torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype, device=device)
    X_init_expected = X_init_actual.clone()
    Y_init = torch.tensor([[1.0]], dtype=dtype, device=device)
    bounds0 = torch.zeros(2, 3, device=device, dtype=dtype)
    bounds0[1] = 1.0
    dh.save_init(X_init_actual=X_init_actual, X_init_expected=X_init_expected, Y_init=Y_init, bounds=bounds0)
    return dh


def test_penalty_mask_ellipsoid_excludes_inside_points(torch, tmp_run_dir):
    from src.utils.simplex import composition_to_ilr, ilr_to_composition

    dh = _make_minimal_datahandler(torch, tmp_run_dir)
    device = dh.device
    dtype = dh.dtype

    d = 3
    needle = torch.tensor([0.2, 0.3, 0.5], dtype=dtype, device=device)
    # ILR-space ellipsoid: M = identity => unit ball in ILR delta-space.
    # B=None signals ILR mode.
    M = torch.eye(d - 1, dtype=dtype, device=device)

    dh.add_needle(
        needle=needle,
        needle_value=1.0,
        needle_penalty_radius=0.0,
        activation=0,
        zoom=0,
        iteration=0,
        M=M,
        B=None,
    )

    # The needle itself has delta_z = 0 -> quad = 0 <= 1 -> inside (penalized).
    mask = dh.get_penalty_mask(needle.unsqueeze(0))
    assert mask.shape == (1,)
    assert mask.item() is False

    # Move 2 units along ILR coordinate 0: delta_z = [2, 0], quad = 4 > 1 -> outside.
    needle_ilr = composition_to_ilr(needle.unsqueeze(0)).squeeze(0)
    delta_z = torch.tensor([2.0, 0.0], dtype=dtype, device=device)
    z_out = needle_ilr + delta_z
    x_out = ilr_to_composition(z_out.unsqueeze(0), d).squeeze(0)
    mask2 = dh.get_penalty_mask(x_out.unsqueeze(0))
    assert mask2.item() is True



