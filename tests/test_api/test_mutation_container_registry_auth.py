"""Tests for GraphQL registry credential operations."""

from runpod.api.mutations.container_register_auth import (
    update_container_registry_auth,
)


def test_update_container_registry_auth():
    mutation = update_container_registry_auth("registry", "user", "password")

    assert "mutation UpdateRegistryAuth" in mutation
    assert 'id: "registry"' in mutation
    assert 'username: "user"' in mutation
    assert 'password: "password"' in mutation
