import os
import jax
import jax.numpy as jnp


def test_backend(backend_name: str):
    print("="*40)
    print(f"Testing backend: {backend_name}")
    os.environ["JAX_DIST_BACKEND"] = backend_name

    try:
        x = jnp.arange(10)
        y = x * 2
        print("Computation result:", y)
        print("Devices:", jax.devices())
        print(f"{backend_name} works.")
    except Exception as e:
        print(f"{backend_name} failed:", e)

if __name__ == "__main__":
    for backend in ["gloo", "nccl", "mpi"]:
        test_backend(backend)
