import jax
import jax.numpy as jnp
from jax import random


def main():
    print("Testing JAX on CPU...")
    print(f"Devices: {jax.devices()}")
    
    key = random.PRNGKey(0)
    x = random.normal(key, (100, 1))
    y = 3 * x.squeeze() + 2 + 0.1 * random.normal(key, (100,))
    
    params = {
        'w': jnp.array(0.0),
        'b': jnp.array(0.0)
    }
    
    def loss_fn(params, x, y):
        pred = params['w'] * x + params['b']
        return jnp.mean((pred - y) ** 2)
    
    grad_fn = jax.grad(loss_fn)
    
    learning_rate = 0.1
    for i in range(100):
        grads = grad_fn(params, x.squeeze(), y)
        params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
        
        if i % 20 == 0:
            loss = loss_fn(params, x.squeeze(), y)
            print(f"Iteration {i}, loss: {loss:.4f}, w: {params['w']:.3f}, b: {params['b']:.3f}")
    
    print("Training completed!")
    print(f"Final parameters: w = {params['w']:.3f}, b = {params['b']:.3f}")

if __name__ == "__main__":
    main()
