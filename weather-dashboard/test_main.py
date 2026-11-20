"""Run through functions-framework's real app: the /robots.txt fix depends on
that app's routing, which a bare call to the handler would bypass."""

import pathlib

import functions_framework
import pytest
import requests

SOURCE = str(pathlib.Path(__file__).with_name("main.py"))


@pytest.fixture
def client(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("unexpected call to get-weather-data")

    monkeypatch.setattr(requests, "post", no_network)
    app = functions_framework.create_app(target="display_weather_data", source=SOURCE)
    return app.test_client()


def test_robots_txt_disallows_all_without_calling_the_api(client):
    resp = client.get("/robots.txt")
    assert resp.status_code == 200
    assert resp.mimetype == "text/plain"
    assert resp.get_data(as_text=True) == "User-agent: *\nDisallow: /\n"


def test_other_paths_still_reach_the_inner_flask_app(client):
    # A static asset is served by the inner app without any API call.
    resp = client.get("/static/common_navbar.css")
    assert resp.status_code == 200
    assert client.get("/no-such-page").status_code == 404
