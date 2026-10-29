import json
import unittest
from pathlib import Path

from aequitas.flow.plots.pareto import visualize


class TestParetoVisualize(unittest.TestCase):
    def test_make_html(self):
        payload = {"models": [{"model_id": 0, "Predictive Equality": 0.9}]}
        html = visualize._make_html(payload)

        bundle = Path(visualize.__file__).parent / "js" / "dist" / "fairAutoML.js"
        self.assertIn(bundle.read_text()[:500], html)
        self.assertIn(json.dumps(payload), html)
        self.assertNotIn("$library_bundle", html)
        self.assertNotIn("$payload", html)
