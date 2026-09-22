"""Check the extension files ComfyUI discovers without importing model runtimes."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendAssetsTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        declarations = [
            statement.value
            for statement in tree.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "WEB_DIRECTORY"
                for target in statement.targets
            )
        ]
        self.assertEqual(len(declarations), 1, "Declare WEB_DIRECTORY only once")
        self.web_root = ROOT / ast.literal_eval(declarations[0])

    def test_all_extensions_are_discoverable(self):
        # server.py recursively lists JavaScript only beneath WEB_DIRECTORY.
        discovered = {path.name for path in self.web_root.rglob("*.js")}
        self.assertTrue(
            {"node_id_editor.js", "gpt_reverse_prompt.js", "output_browser.js"}
            <= discovered,
            f"Missing frontend extensions: {discovered}",
        )

    def test_output_browser_styles_are_served_beside_script(self):
        script = next(self.web_root.rglob("output_browser.js"))
        self.assertTrue(script.with_suffix(".css").is_file())
        self.assertIn(
            'new URL("./output_browser.css", import.meta.url)',
            script.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
