# tests/test_ui_pages.py
# ---------------------
# Unit tests for checking UI HTML page template routing.

import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


class TestUIPages(unittest.TestCase):
    """
    Test suite verifying that HTML static/Jinja2 page endpoints resolve correctly.
    """
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_dashboard_page_resolves(self):
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("Workspace Dashboard", res.text)

    def test_conversations_page_resolves(self):
        res = self.client.get("/conversations")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("Conversations", res.text)

    def test_collections_page_resolves(self):
        res = self.client.get("/collections")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("Document Collections", res.text)

    def test_documents_page_resolves(self):
        res = self.client.get("/documents")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("Document Management", res.text)

    def test_workspaces_page_resolves(self):
        res = self.client.get("/workspaces")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("Workspaces", res.text)

    def test_search_history_page_resolves(self):
        res = self.client.get("/search-history")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("Search History & Analytics", res.text)


if __name__ == "__main__":
    unittest.main()
