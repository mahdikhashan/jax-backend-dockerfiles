import os
import jax
import jax.numpy as jnp
from jax import lax, pmap
from jax.experimental import multihost_utils
import jax.distributed as dist

_initialized = False  # track ourselves


def init_dist():
    global _initialized
    nproc = int(os.environ.get("JAX_NUM_PROCESSES", "1"))
    if nproc > 1 and not _initialized:
        coord = os.environ["JAX_COORDINATOR_ADDRESS"]   # e.g. "127.0.0.1:1234"
        pid = int(os.environ["JAX_PROCESS_ID"])         # 0..nproc-1
        dist.initialize(
            coordinator_address=coord,
            num_processes=nproc,
            process_id=pid,
        )
        multihost_utils.sync_global_devices("after initialize")
        _initialized = True


def main():
    init_dist()

    proc_id = dist.process_index() if _initialized else 0
    nproc   = dist.process_count() if _initialized else 1
    print(f"[proc {proc_id}/{nproc}] backend={jax.default_backend()} "
          f"local_devices={jax.local_device_count()} global_devices={jax.device_count()}")

    def f(x):
        y = x + 1
        s = lax.psum(y, axis_name="i")
        return y, s

    xs = jnp.arange(jax.local_device_count(), dtype=jnp.int32)
    ys, sums = pmap(f, axis_name="i")(xs)

    all_xs   = multihost_utils.process_allgather(xs)
    all_ys   = multihost_utils.process_allgather(ys)
    all_sums = multihost_utils.process_allgather(sums)

    if proc_id == 0:
        print("per-device x:", all_xs)
        print("per-device y=x+1:", all_ys)
        print("per-device psum(y):", all_sums)

    expected = jnp.sum(jnp.arange(jax.device_count(), dtype=jnp.int32) + 1)
    if _initialized:
        expected = multihost_utils.broadcast_one_to_all(expected)

    ok = bool(sums[0] == expected)
    print(f"[proc {proc_id}] OK={ok}, expected psum={int(expected)}")


if __name__ == "__main__":
    main()
