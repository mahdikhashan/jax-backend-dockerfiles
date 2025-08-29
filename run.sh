docker run --rm -it \
  --network host \
  -e JAX_PLATFORMS=cpu \
  -e XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  -v "$(pwd)":/workspace \
  jax-multi-backend \
  python3 test_multi_host.py
