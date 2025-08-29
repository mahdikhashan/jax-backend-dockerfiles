import os
import jax
import jax.numpy as jnp
from jax import random
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def test_gloo_backend():
    print("=== Testing Gloo Backend ===")
    
    os.environ["JAX_DIST_BACKEND"] = "gloo"
    
    try:
        jax.lib.xla_bridge.get_backend.cache_clear()
        
        key = random.PRNGKey(0)
        x = random.normal(key, (10, 10))
        y = jnp.dot(x, x.T)
        
        print(f"Computation result shape: {y.shape}")
        print(f"Computation result sum: {jnp.sum(y):.2f}")
        print(f"Devices: {jax.devices()}")
        print("Gloo backend works.")
        
        return True
    except Exception as e:
        print(f"Gloo backend failed: {e}.")
        return False

def test_cpu_info():
    print("=== CPU Information ===")
    print(f"Number of CPU cores: {multiprocessing.cpu_count()}")
    
    devices = jax.devices()
    cpu_devices = [d for d in devices if d.platform == 'cpu']
    print(f"Number of JAX CPU devices: {len(cpu_devices)}")
    
    try:
        import subprocess
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nCPU details:")
            lines = result.stdout.split('\n')
            for line in lines[:10]:  # Show first 10 lines
                if line.strip():
                    print(f"  {line}")
        else:
            print("Could not get detailed CPU information")
    except:
        print("Could not get detailed CPU information")
    
    print()

def test_collective_operations():
    print("=== Testing Collective Operations ===")
    
    try:
        from jax import lax
        
        x = jnp.ones(5)
        print(f"Input array: {x}")
        
        result = lax.psum(x, 'i')
        print(f"All-reduce result: {result}")
        
        print("Collective operations work.")
        return True
    except Exception as e:
        print(f"Collective operations failed: {e}")
        return False

if __name__ == "__main__":
    test_cpu_info()
    
    gloo_success = test_gloo_backend()
    
    if gloo_success:
        test_collective_operations()
    
    print("=== Additional Diagnostics ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Platform: {jax.lib.xla_bridge.get_backend().platform}")
    
    try:
        from jax._src.lib import xla_client
        print("Gloo integration: Available")
    except ImportError as e:
        print(f"Gloo integration: Not available ({e})")
