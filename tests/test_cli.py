import tempfile
import unittest
from pathlib import Path

from contract_qa.cli import discover_documents


class DiscoverDocumentsTests(unittest.TestCase):
    def test_returns_only_pdf_files_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            (raw_dir / "b.pdf").touch()
            (raw_dir / "a.PDF").touch()
            (raw_dir / "notes.txt").touch()
            (raw_dir / "nested").mkdir()
            (raw_dir / "nested" / "c.pdf").touch()

            self.assertEqual(
                [path.name for path in discover_documents(raw_dir)],
                ["a.PDF", "b.pdf"],
            )

    def test_missing_directory_returns_empty_list(self) -> None:
        self.assertEqual(discover_documents(Path("does-not-exist")), [])


if __name__ == "__main__":
    unittest.main()
