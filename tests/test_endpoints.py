"""Tests of the endpoint check that keeps requests away from the server's own network."""

from __future__ import annotations

import unittest
from unittest import mock

import httpx

from app import endpoints


class CheckUrlTest(unittest.TestCase):
    def assertRefused(self, url):
        with self.assertRaises(endpoints.EndpointNotAllowed):
            endpoints.check_url(url)

    def test_internal_addresses_are_refused(self):
        for url in ("http://127.0.0.1:8000/v1", "http://localhost/v1", "http://10.1.2.3/v1",
                    "http://192.168.0.5:11434/v1", "http://169.254.169.254/latest", "http://[::1]/v1",
                    "http://[::ffff:127.0.0.1]/v1", "http://0.0.0.0/v1", "http://100.64.0.1/v1"):
            with self.subTest(url=url):
                self.assertRefused(url)

    def test_other_schemes_are_refused(self):
        for url in ("file:///etc/passwd", "ftp://141.76.18.84/", "llm.scads.ai/v1", ""):
            with self.subTest(url=url):
                self.assertRefused(url)

    def test_public_address_is_allowed(self):
        endpoints.check_url("https://141.76.18.84/v1")

    def test_can_be_turned_off(self):
        with mock.patch.object(endpoints, "ALLOW_PRIVATE", True):
            endpoints.check_url("http://127.0.0.1:8000/v1")

    def test_every_request_is_checked(self):
        """Also the target of a redirect from a public endpoint."""
        def handler(request):
            if request.url.host == "141.76.18.84":
                return httpx.Response(302, headers={"Location": "http://127.0.0.1:8000/secret"})
            return httpx.Response(200, text="internal")

        client = endpoints.http_client()
        client._transport = httpx.MockTransport(handler)
        with self.assertRaises(endpoints.EndpointNotAllowed):
            client.get("https://141.76.18.84/v1/models")


if __name__ == "__main__":
    unittest.main()
