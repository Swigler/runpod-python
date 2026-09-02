"""Tests for the API wrapper commands."""

from unittest.mock import patch

import pytest

from runpod.api import ctl_commands
from runpod.error import QueryError


def test_get_user_uses_graphql():
    with patch(
        "runpod.api.ctl_commands.run_graphql_query",
        return_value={"data": {"myself": {"id": "user"}}},
    ) as request:
        assert ctl_commands.get_user(api_key="key") == {"id": "user"}

    request.assert_called_once_with(ctl_commands.user_queries.QUERY_USER, api_key="key")


def test_update_user_settings_replaces_ssh_keys():
    with (
        patch("runpod.api.ctl_commands.run_rest_request") as request,
        patch(
            "runpod.api.ctl_commands.get_user",
            return_value={"id": "user", "pubKey": "ssh-ed25519 key user"},
        ) as get_user,
    ):
        result = ctl_commands.update_user_settings(
            "\nssh-ed25519 key user\n\n", api_key="key"
        )

    assert result == {"id": "user", "pubKey": "ssh-ed25519 key user"}
    request.assert_called_once_with(
        "PUT",
        "/v2/account/ssh-keys",
        api_key="key",
        json={"keys": ["ssh-ed25519 key user"]},
    )
    get_user.assert_called_once_with(api_key="key")


def test_get_gpus_unwraps_response():
    gpus = [{"id": "NVIDIA A100", "name": "A100", "memory": 80}]
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"gpus": gpus}
    ) as request:
        assert ctl_commands.get_gpus(api_key="key") == gpus

    request.assert_called_once_with("GET", "/v2/catalog/gpus", api_key="key")


def test_get_gpu_requests_pod_availability():
    gpu = {"id": "NVIDIA A100"}
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value=gpu
    ) as request:
        assert ctl_commands.get_gpu("NVIDIA A100", 2, api_key="key") == gpu

    request.assert_called_once_with(
        "GET",
        "/v2/catalog/gpus/NVIDIA%20A100",
        api_key="key",
        params={"include": "AVAILABILITY", "product": "POD", "count": 2},
    )


def test_get_gpu_converts_not_found_to_value_error():
    with (
        patch(
            "runpod.api.ctl_commands.run_rest_request",
            side_effect=QueryError("not found", status_code=404),
        ),
        pytest.raises(ValueError, match="No GPU found"),
    ):
        ctl_commands.get_gpu("missing")


def test_get_gpu_propagates_other_api_errors():
    with (
        patch(
            "runpod.api.ctl_commands.run_rest_request",
            side_effect=QueryError("forbidden", status_code=403),
        ),
        pytest.raises(QueryError, match="forbidden"),
    ):
        ctl_commands.get_gpu("NVIDIA A100")


def test_get_pods_unwraps_response():
    pods = [{"id": "pod"}]
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"pods": pods}
    ) as request:
        assert ctl_commands.get_pods(api_key="key") == pods

    request.assert_called_once_with("GET", "/v2/pods", api_key="key")


def test_get_pod_escapes_id():
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"id": "pod/id"}
    ) as request:
        assert ctl_commands.get_pod("pod/id", api_key="key") == {"id": "pod/id"}

    request.assert_called_once_with("GET", "/v2/pods/pod%2Fid", api_key="key")


def test_get_pod_returns_none_when_not_found():
    with patch(
        "runpod.api.ctl_commands.run_rest_request",
        side_effect=QueryError("not found", status_code=404),
    ):
        assert ctl_commands.get_pod("missing") is None


def test_get_pod_propagates_other_api_errors():
    with (
        patch(
            "runpod.api.ctl_commands.run_rest_request",
            side_effect=QueryError("forbidden", status_code=403),
        ),
        pytest.raises(QueryError, match="forbidden"),
    ):
        ctl_commands.get_pod("pod")


def test_create_gpu_pod_translates_request():
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"id": "pod"}
    ) as request:
        result = ctl_commands.create_pod(
            name="training",
            image_name="runpod/pytorch:latest",
            gpu_type_id="NVIDIA A100",
            cloud_type="COMMUNITY",
            start_ssh=True,
            data_center_id="US-KS-2",
            gpu_count=2,
            volume_in_gb=20,
            container_disk_in_gb=50,
            docker_args="python main.py",
            ports="8888/http, 22/tcp",
            volume_mount_path="/workspace",
            env={"COUNT": 2},
            allowed_cuda_versions=["12.8", "12.6"],
        )

    assert result == {"id": "pod"}
    request.assert_called_once_with(
        "POST",
        "/v2/pods",
        json={
            "name": "training",
            "image": "runpod/pytorch:latest",
            "args": "python main.py",
            "startSsh": True,
            "cloud": "COMMUNITY",
            "dataCenterIds": ["US-KS-2"],
            "disk": 50,
            "ports": ["8888/http", "22/tcp"],
            "env": {"COUNT": "2"},
            "mounts": {
                "persistent": {"size": 20, "path": "/workspace"}
            },
            "gpu": {
                "id": "NVIDIA A100",
                "count": 2,
                "allowedCudaVersions": ["12.8", "12.6"],
            },
        },
    )


def test_create_gpu_pod_with_network_volume_and_template():
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"id": "pod"}
    ) as request:
        ctl_commands.create_pod(
            name="training",
            template_id="template",
            gpu_type_id="NVIDIA A100",
            network_volume_id="volume",
        )

    body = request.call_args.kwargs["json"]
    assert body["templateId"] == "template"
    assert body["mounts"] == {
        "network": [{"volumeId": "volume", "path": "/runpod-volume"}]
    }
    assert "image" not in body
    assert "disk" not in body
    assert "cloud" not in body


def test_create_cpu_pod_translates_instance_id():
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"id": "pod"}
    ) as request:
        ctl_commands.create_pod(
            "cpu-pod", "python:3.11", instance_id="cpu3c-4-8"
        )

    assert request.call_args.kwargs["json"]["cpu"] == {
        "id": "cpu3c",
        "vcpuCount": 4,
    }


def test_create_cpu_pod_requires_instance_id():
    with pytest.raises(ValueError, match="instance_id"):
        ctl_commands.create_pod("cpu-pod", "python:3.11")


def test_create_cpu_pod_validates_instance_id():
    with pytest.raises(ValueError, match="format"):
        ctl_commands.create_pod(
            "cpu-pod", "python:3.11", instance_id="cpu3c-invalid"
        )


def test_create_pod_validates_image_and_cloud():
    with pytest.raises(ValueError, match="Either image_name or template_id"):
        ctl_commands.create_pod("pod", gpu_type_id="NVIDIA A100")

    with pytest.raises(ValueError, match="cloud_type"):
        ctl_commands.create_pod(
            "pod", "image", gpu_type_id="NVIDIA A100", cloud_type="INVALID"
        )


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"support_public_ip": False}, "support_public_ip"),
        ({"country_code": "US"}, "country_code"),
        ({"min_vcpu_count": 8}, "min_vcpu_count"),
        ({"min_memory_in_gb": 32}, "min_memory_in_gb"),
        ({"min_download": 100}, "min_download"),
        ({"min_upload": 100}, "min_upload"),
    ],
)
def test_create_gpu_pod_rejects_unsupported_constraints(kwargs, field):
    with pytest.raises(ValueError, match=field):
        ctl_commands.create_pod(
            "pod", "image", gpu_type_id="NVIDIA A100", **kwargs
        )


def test_stop_and_resume_pod_use_actions():
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"id": "pod"}
    ) as request:
        assert ctl_commands.stop_pod("pod") == {"id": "pod"}
        assert ctl_commands.resume_pod("pod", 8) == {"id": "pod"}

    assert request.call_args_list[0].args == (
        "POST",
        "/v2/pods/pod/action",
    )
    assert request.call_args_list[0].kwargs == {"json": {"action": "stop"}}
    assert request.call_args_list[1].kwargs == {"json": {"action": "start"}}


def test_terminate_pod_deletes_resource():
    with patch("runpod.api.ctl_commands.run_rest_request", return_value=None) as request:
        assert ctl_commands.terminate_pod("pod/id") is None

    request.assert_called_once_with("DELETE", "/v2/pods/pod%2Fid")


def test_create_template_translates_request():
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"id": "template"}
    ) as request:
        result = ctl_commands.create_template(
            name="template",
            image_name="image",
            docker_start_cmd="python main.py",
            container_disk_in_gb=20,
            volume_in_gb=50,
            ports="8888/http,22/tcp",
            env={"PORT": 8888},
            is_serverless=True,
            registry_auth_id="registry",
        )

    assert result == {"id": "template"}
    request.assert_called_once_with(
        "POST",
        "/v2/templates",
        json={
            "name": "template",
            "image": "image",
            "args": "python main.py",
            "disk": 20,
            "mounts": {
                "persistent": {"size": 50, "path": "/workspace"}
            },
            "ports": ["8888/http", "22/tcp"],
            "env": {"PORT": "8888"},
            "serverless": True,
            "registry": "registry",
        },
    )


def test_get_endpoints_unwraps_response():
    endpoints = [{"id": "endpoint"}]
    with patch(
        "runpod.api.ctl_commands.run_rest_request",
        return_value={"endpoints": endpoints},
    ) as request:
        assert ctl_commands.get_endpoints() == endpoints

    request.assert_called_once_with("GET", "/v2/serverless")


def test_create_endpoint_translates_queue_scaling():
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"id": "endpoint"}
    ) as request:
        result = ctl_commands.create_endpoint(
            name="endpoint",
            template_id="template",
            gpu_ids="AMPERE_16,ADA_24",
            network_volume_id="volume",
            locations="US-KS-2, EU-RO-1",
            idle_timeout=10,
            scaler_value=8,
            workers_min=1,
            workers_max=5,
            flashboot=True,
            allowed_cuda_versions="12.8,12.6",
            gpu_count=2,
        )

    assert result == {"id": "endpoint"}
    request.assert_called_once_with(
        "POST",
        "/v2/serverless",
        json={
            "name": "endpoint",
            "templateId": "template",
            "type": "QUEUE",
            "gpu": {
                "pools": ["AMPERE_16", "ADA_24"],
                "count": 2,
                "allowedCudaVersions": ["12.8", "12.6"],
            },
            "networkVolumes": ["volume"],
            "dataCenterIds": ["US-KS-2", "EU-RO-1"],
            "workers": {"min": 1, "max": 5, "idleTimeout": 10},
            "scaling": {"type": "QUEUE_DELAY", "queueDelay": 8},
            "flashboot": "FLASHBOOT",
        },
    )


def test_create_endpoint_translates_worker_count_scaling():
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"id": "endpoint"}
    ) as request:
        ctl_commands.create_endpoint(
            "endpoint", "template", scaler_type="WORKER_COUNT", scaler_value=2
        )

    body = request.call_args.kwargs["json"]
    assert body["scaling"] == {"type": "REQUEST_COUNT", "requestCount": 2}
    assert body["workers"] == {"min": 0, "max": 3}


def test_create_endpoint_rejects_invalid_scaler():
    with pytest.raises(ValueError, match="scaler_type"):
        ctl_commands.create_endpoint(
            "endpoint", "template", scaler_type="INVALID"
        )


def test_update_endpoint_template_uses_patch():
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"id": "endpoint"}
    ) as request:
        assert ctl_commands.update_endpoint_template("endpoint/id", "template") == {
            "id": "endpoint"
        }

    request.assert_called_once_with(
        "PATCH",
        "/v2/serverless/endpoint%2Fid",
        json={"templateId": "template"},
    )


def test_create_container_registry_auth_uses_rest():
    with patch(
        "runpod.api.ctl_commands.run_rest_request", return_value={"id": "registry"}
    ) as request:
        result = ctl_commands.create_container_registry_auth(
            "registry", "user", "password"
        )

    assert result == {"id": "registry"}
    request.assert_called_once_with(
        "POST",
        "/v2/registries",
        json={"name": "registry", "username": "user", "password": "password"},
    )


def test_update_container_registry_auth_uses_graphql():
    with patch(
        "runpod.api.ctl_commands.run_graphql_query",
        return_value={"data": {"updateRegistryAuth": {"id": "registry"}}},
    ) as request:
        result = ctl_commands.update_container_registry_auth(
            "registry", "user", "password"
        )

    assert result == {"id": "registry"}
    mutation = request.call_args.args[0]
    assert "mutation UpdateRegistryAuth" in mutation
    assert 'id: "registry"' in mutation


def test_delete_container_registry_auth_uses_rest():
    with patch("runpod.api.ctl_commands.run_rest_request", return_value=None) as request:
        assert ctl_commands.delete_container_registry_auth("registry/id") is True

    request.assert_called_once_with("DELETE", "/v2/registries/registry%2Fid")
