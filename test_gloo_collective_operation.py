import os
import jax
import jax.numpy as jnp
from jax import random
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["JAX_DIST_BACKEND"] = "gloo"

def test_basic_operations():
    """Test basic JAX operations with Gloo backend"""
    print("=== Testing Basic Operations ===")
    
    key = random.PRNGKey(0)
    x = random.normal(key, (5, 5))
    y = jnp.dot(x, x.T)
    
    print(f"Computation result shape: {y.shape}")
    print(f"Computation result sum: {jnp.sum(y):.2f}")
    print(f"Devices: {jax.devices()}")
    print("✅ Basic operations work\n")

def test_collective_operations():
    """Test collective operations using pmap"""
    print("=== Testing Collective Operations ===")
    
    try:
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
        
        backend = jax.lib.xla_bridge.get_backend()
        print(f"Available devices: {jax.devices()}")
        
        def sum_all_devices(x):
            return jax.lax.psum(x, 'i')
        
        pmapped_func = jax.pmap(sum_all_devices, axis_name='i')
        
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2 devices, 3 values)
        print(f"Input: {x}")
        
        result = pmapped_func(x)
        print(f"Result after psum: {result}")
        
        print("✅ Collective operations work\n")
        return True
        
    except Exception as e:
        print(f"❌ Collective operations failed: {e}\n")
        return False

def test_system_info():
    """Display system information"""
    print("=== System Information ===")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    
    try:
        backend = jax.extend.backend.get_backend()
        print(f"Platform: {backend.platform}")
    except:
        backend = jax.lib.xla_bridge.get_backend()
        print(f"Platform: {backend.platform}")
    print()

if __name__ == "__main__":
    test_system_info()
    test_basic_operations()
    test_collective_operations()
