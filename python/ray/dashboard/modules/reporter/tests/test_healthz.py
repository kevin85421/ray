import sys

import pytest
from ray._common.test_utils import wait_for_condition
import requests

import ray._private.ray_constants as ray_constants
from ray._private.test_utils import find_free_port
from ray.tests.conftest import *  # noqa: F401 F403


def test_healthz_head(monkeypatch, ray_start_cluster):
    dashboard_port = find_free_port()
    h = ray_start_cluster.add_node(dashboard_port=dashboard_port)
    uri = f"http://localhost:{dashboard_port}/api/gcs_healthz"
    wait_for_condition(lambda: requests.get(uri).status_code == 200)
    h.all_processes[ray_constants.PROCESS_TYPE_GCS_SERVER][0].process.kill()
    # It'll either timeout or just return an error
    try:
        wait_for_condition(lambda: requests.get(uri, timeout=1) != 200, timeout=4)
    except RuntimeError as e:
        assert "Read timed out" in str(e)


def test_healthz_agent_1(monkeypatch, ray_start_cluster):
    agent_port = find_free_port()
    h = ray_start_cluster.add_node(dashboard_agent_listen_port=agent_port)
    uri = f"http://localhost:{agent_port}/api/local_raylet_healthz"

    wait_for_condition(lambda: requests.get(uri).status_code == 200)

    h.all_processes[ray_constants.PROCESS_TYPE_GCS_SERVER][0].process.kill()
    # GCS's failure will not lead to healthz failure
    assert requests.get(uri).status_code == 200


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSTOP only on posix")
def test_healthz_agent_2(monkeypatch, ray_start_cluster):
    monkeypatch.setenv("RAY_health_check_failure_threshold", "3")
    monkeypatch.setenv("RAY_health_check_timeout_ms", "100")
    monkeypatch.setenv("RAY_health_check_period_ms", "1000")
    monkeypatch.setenv("RAY_health_check_initial_delay_ms", "0")

    agent_port = find_free_port()
    h = ray_start_cluster.add_node(dashboard_agent_listen_port=agent_port)
    uri = f"http://localhost:{agent_port}/api/local_raylet_healthz"

    wait_for_condition(lambda: requests.get(uri).status_code == 200)

    import signal

    h.all_processes[ray_constants.PROCESS_TYPE_RAYLET][0].process.send_signal(
        signal.SIGSTOP
    )

    # GCS still think raylet is alive.
    assert requests.get(uri).status_code == 200
    # But after heartbeat timeout, it'll think the raylet is down.
    wait_for_condition(lambda: requests.get(uri).status_code != 200)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
