import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: requires huggingface.co reachability")
    config.addinivalue_line(
        "markers", "slow: compiles/runs the C++ reader proof")


@pytest.fixture(autouse=True)
def _skip_network_without_flag(request):
    if request.node.get_closest_marker("network"):
        import urllib.request
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    "https://huggingface.co",
                    headers={"Range": "bytes=0-0"}), timeout=10)
        except Exception:
            pytest.skip("huggingface.co unreachable")
