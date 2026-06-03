import json
from pathlib import Path


def test_datahandler_all_methods_smoke(torch, tmp_run_dir):
    """
    Smoke-test every method on DataHandler at least once.
    This is not a deep behavioral test; it ensures functions are callable
    and their basic contracts hold.
    """
    from src.utils.datahandler import DataHandler

    device = torch.device("cuda")
    dtype = torch.float64

    dh = DataHandler(
        directory=str(tmp_run_dir),
        device="cuda",
        dtype=dtype,
        d=3,
        max_snapshots=5,
    )

    # _init_storage is called in __init__, but also directly callable
    dh._init_storage()

    X_init_actual = torch.tensor([[0.2, 0.3, 0.5]], device=device, dtype=dtype)
    X_init_expected = X_init_actual.clone()
    Y_init = torch.tensor([[1.0]], device=device, dtype=dtype)
    bounds0 = torch.zeros(2, 3, device=device, dtype=dtype)
    bounds0[1] = 1.0

    dh.save_init(X_init_actual=X_init_actual, X_init_expected=X_init_expected, Y_init=Y_init, bounds=bounds0)
    dh._save_config()

    # Iteration state
    dh.update_iteration_state(activation=0, zoom=0, iteration=0, no_improvements=0)
    a, z, it, ni = dh.get_iteration_state()
    assert (a, z, it, ni) == (0, 0, 0, 0)

    # Add some data points
    X_new = torch.tensor([[0.25, 0.25, 0.5], [0.1, 0.2, 0.7]], device=device, dtype=dtype)
    Y_new = torch.tensor([[0.9], [0.8]], device=device, dtype=dtype)
    dh.add_all_points(new_X_actual=X_new, new_X_expected=X_new.clone(), new_Y=Y_new)

    X_all_actual, X_all_expected, Y_all = dh.get_all_points()
    assert X_all_actual.shape[0] >= 3
    assert Y_all.shape[0] == X_all_actual.shape[0]

    # Penalty mask paths
    pm_all = dh.get_penalty_mask()
    assert pm_all.dtype == torch.bool
    pm_pts = dh.get_penalty_mask(X_new)
    assert pm_pts.shape == (X_new.shape[0],)

    X_gp, Y_gp = dh.get_gp_data()
    assert X_gp.shape[0] == Y_gp.shape[0]

    best_X, best_Y, best_idx = dh.get_best_unpenalized()
    assert best_X is not None and best_Y is not None and best_idx is not None

    # Bounds computations — now returns a (2, d) tensor
    nb = dh.determine_new_bounds()
    assert isinstance(nb, torch.Tensor)
    assert nb.shape == (2, 3)

    # Add a needle (sphere fallback) and then demote to old
    dh.add_needle(
        needle=best_X,
        needle_value=float(best_Y.item()),
        needle_penalty_radius=0.0,
        activation=0,
        zoom=0,
        iteration=0,
        M=None,
        B=None,
    )
    _ = dh.get_needle_locations()
    _ = dh.get_needle_results()
    _ = dh.get_needles_and_penalty_radii()
    _ = dh.get_needle_ellipsoids()

    # Snapshotting / loading
    dh.take_snapshot("smoke", activation=0, zoom=0, iteration=0)
    dh.push_checkpoint("smoke_ckpt", is_permanent=False)
    dh._cleanup_old_snapshots()

    # load_state should load from latest snapshot
    la, lz, lit, lni = dh.load_state()
    assert isinstance(la, int) and isinstance(lz, int) and isinstance(lit, int) and isinstance(lni, int)

    # _load_from_snapshot: point it to latest.txt contents
    latest_path = Path(dh.run_dir) / "latest.txt"
    if latest_path.exists():
        snap_name = latest_path.read_text().strip()
        dh._load_from_snapshot(snap_name)

    # _load_from_old_checkpoint: should safely no-op if missing
    dh._load_from_old_checkpoint("does_not_exist")

    # _load_tensors: round-trip current tensors.pt
    snap_dir = Path(dh.run_dir) / "snapshots"
    if snap_dir.exists():
        snaps = sorted([p for p in snap_dir.iterdir() if p.is_dir()])
        if snaps:
            tensors = torch.load(snaps[-1] / "tensors.pt", map_location="cuda")
            dh._load_tensors(tensors)

            # _load_needles_json: create minimal and load
            needles_json = snaps[-1] / "needles.json"
            if needles_json.exists():
                dh._load_needles_json(needles_json)
            else:
                needles_json.write_text(json.dumps([]), encoding="utf-8")
                dh._load_needles_json(needles_json)

    # Noise helpers
    _ = dh.get_input_noise()
    _ = dh.get_normalized_input_noise()

