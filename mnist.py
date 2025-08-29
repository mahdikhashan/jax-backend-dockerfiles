import os
import jax
import jax.numpy as jnp
import optax
from jax import random
from flax import linen as nn
from flax.training import train_state
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels).mean()


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {"loss": loss, "accuracy": accuracy}


def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch["image"])
        loss = cross_entropy_loss(logits, batch["label"])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name="devices")

    state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits, batch["label"])

    return state, metrics


def eval_step(state, batch):
    logits = state.apply_fn(state.params, batch["image"])
    return compute_metrics(logits, batch["label"])


def get_datasets(batch_size=128):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def convert_batch(batch):
    images, labels = batch
    return {"image": jnp.array(images.numpy()), "label": jnp.array(labels.numpy())}


def main():
    print("Initializing training with emulated devices...")

    rng = random.PRNGKey(0)

    train_loader, test_loader = get_datasets()

    model = MLP()
    rng, init_rng = random.split(rng)
    params = model.init(init_rng, jnp.ones([1, 28, 28, 1]))["params"]

    tx = optax.adam(learning_rate=1e-3)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    num_devices = jax.local_device_count()
    print(f"Number of emulated devices: {num_devices}")

    state = jax.device_put_replicated(state, jax.local_devices())

    p_train_step = jax.pmap(train_step, axis_name="devices")
    p_eval_step = jax.pmap(eval_step, axis_name="devices")

    num_epochs = 3
    for epoch in range(num_epochs):
        train_metrics = []
        for i, batch in enumerate(train_loader):
            if i >= 10:
                break

            jax_batch = convert_batch(batch)

            batch_size = jax_batch["image"].shape[0]
            if batch_size % num_devices != 0:
                pad_size = num_devices - (batch_size % num_devices)
                jax_batch["image"] = jnp.concatenate(
                    [jax_batch["image"], jnp.zeros((pad_size, 28, 28, 1))], axis=0
                )
                jax_batch["label"] = jnp.concatenate(
                    [jax_batch["label"], jnp.zeros(pad_size, dtype=jnp.int32)], axis=0
                )
                batch_size += pad_size

            split_size = batch_size // num_devices
            split_batch = {
                "image": jax_batch["image"].reshape(
                    (num_devices, split_size, 28, 28, 1)
                ),
                "label": jax_batch["label"].reshape((num_devices, split_size)),
            }

            state, metrics = p_train_step(state, split_batch)
            train_metrics.append(metrics)

        if train_metrics:
            avg_train_loss = np.mean([m["loss"][0] for m in train_metrics])
            avg_train_accuracy = np.mean([m["accuracy"][0] for m in train_metrics])

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(
                f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}"
            )

    test_metrics = []
    for i, batch in enumerate(test_loader):
        if i >= 5:
            break

        jax_batch = convert_batch(batch)

        batch_size = jax_batch["image"].shape[0]
        if batch_size % num_devices != 0:
            pad_size = num_devices - (batch_size % num_devices)
            jax_batch["image"] = jnp.concatenate(
                [jax_batch["image"], jnp.zeros((pad_size, 28, 28, 1))], axis=0
            )
            jax_batch["label"] = jnp.concatenate(
                [jax_batch["label"], jnp.zeros(pad_size, dtype=jnp.int32)], axis=0
            )
            batch_size += pad_size

        split_size = batch_size // num_devices
        split_batch = {
            "image": jax_batch["image"].reshape((num_devices, split_size, 28, 28, 1)),
            "label": jax_batch["label"].reshape((num_devices, split_size)),
        }

        metrics = p_eval_step(state, split_batch)
        test_metrics.append(metrics)

    if test_metrics:
        avg_test_loss = np.mean([m["loss"][0] for m in test_metrics])
        avg_test_accuracy = np.mean([m["accuracy"][0] for m in test_metrics])

        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")


if __name__ == "__main__":
    main()
