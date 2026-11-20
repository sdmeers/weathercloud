import pytest
import requests


@pytest.fixture
def client(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("unexpected call to get-weather-data")

    monkeypatch.setattr(requests, "post", no_network)
    import main

    return main.application.test_client()


def test_robots_txt_disallows_all_without_calling_the_api(client):
    resp = client.get("/robots.txt")
    assert resp.status_code == 200
    assert resp.mimetype == "text/plain"
    assert resp.get_data(as_text=True) == "User-agent: *\nDisallow: /\n"


def test_dash_app_still_serves_its_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "_dash-config" in resp.get_data(as_text=True)
