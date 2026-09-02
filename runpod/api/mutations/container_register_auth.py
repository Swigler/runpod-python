"""GraphQL operations for container registry credentials."""


def update_container_registry_auth(
    registry_auth_id: str, username: str, password: str
) -> str:
    """Build the registry credential update mutation."""
    input_dict = {
        "id": registry_auth_id,
        "username": username,
        "password": password,
    }
    input_str = ", ".join(
        f'{key}: "{value}"' for key, value in input_dict.items()
    )

    return f"""
    mutation UpdateRegistryAuth {{
        updateRegistryAuth(input: {{{input_str}}}) {{
            id
            name
        }}
    }}
    """
