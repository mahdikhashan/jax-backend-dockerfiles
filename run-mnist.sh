docker run --rm -it \
  --network host \
  -e JAX_PLATFORMS=cpu \
  -e XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  -v "$(pwd)":/workspace \
  jax-multi-backend \
  bash -c "python3 -m pip install torch torchvision && python3 mnist.py"
