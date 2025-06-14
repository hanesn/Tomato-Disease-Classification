import subprocess
import time
import requests
import pytest
import os
import signal
import sys

def get_creation_flags():
    if os.name == "nt":
        return subprocess.CREATE_NEW_PROCESS_GROUP
    return 0

def wait_until_server_ready(timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            res = requests.get("http://localhost:8000/health")
            if res.status_code in [200, 503]:
                return
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    raise RuntimeError("FastAPI server didn't start in time")

@pytest.fixture(scope="module", autouse=True)
def start_server(request):
    # Decide mode based on marker
    mode = "False"  # default: local
    if "tf_serving" in request.keywords:
        mode = "True"

    env = os.environ.copy()
    env["USE_TF_SERVING"] = mode
    print(f"\n[INFO] Starting server with USE_TF_SERVING={mode}")

    server = subprocess.Popen(
        ["python", "-m", "api.main"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
        creationflags=get_creation_flags()
    )

    try:
        wait_until_server_ready()
        yield
    finally:
        if os.name == "nt":
            server.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            server.terminate()
        server.wait()
        print(f"\n[INFO] Server with USE_TF_SERVING={mode} terminated.")
