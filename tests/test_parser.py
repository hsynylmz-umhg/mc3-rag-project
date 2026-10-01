"""Parser regression checks, including hostile inputs and OCR geometry."""
import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter" / "app"))
import parser


class ParserTests(unittest.TestCase):
    def test_rule_removal_preserves_text_strokes_and_removes_large_frame(self):
        from PIL import Image, ImageDraw
        image = Image.new("L", (800, 600), 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 80, 770, 420), outline=0, width=2)
        draw.rectangle((70, 130, 250, 250), outline=0, width=2)
        draw.line((130, 160, 130, 195), fill=0, width=3)
        clean = parser._remove_rules(image)
        self.assertEqual(clean.getpixel((300, 80)), 255)
        self.assertEqual(clean.getpixel((70, 200)), 255)
        self.assertEqual(clean.getpixel((130, 180)), 0)
        text = Image.new("L", (200, 100), 255)
        ImageDraw.Draw(text).line((100, 10, 100, 90), fill=0, width=4)
        self.assertEqual(parser._remove_rules(text).getpixel((100, 40)), 0)

    def test_hostile_files_do_not_hide_valid_neighbors(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder)
            (corpus / "empty-directory").mkdir()
            (corpus / "a-corrupt.xlsx").write_bytes(b"not a zip")
            (corpus / "b-unknown.bin").write_bytes(b"\x00\xff")
            (corpus / "c-good.csv").write_text("ticket,fixed_in\nBUG-29,7.2.1\n", encoding="utf-8")
            (corpus / "z-good.txt").write_text("Maximum temperature = 88", encoding="utf-8")
            units, errors = parser.parse_corpus(corpus)
            self.assertEqual({u["path"] for u in units}, {"c-good.csv", "z-good.txt"})
            self.assertEqual({e["path"] for e in errors}, {"a-corrupt.xlsx", "b-unknown.bin"})
            self.assertTrue(any("ticket: BUG-29 | fixed_in: 7.2.1" in u["text"] for u in units))

    def test_permission_bits_checked_even_if_root_can_open(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "secret.txt"
            path.write_text("secret", encoding="utf-8")
            info = os.stat_result((stat.S_IFREG, 0, 0, 1, 0, 0, 6, 0, 0, 0))
            with patch.object(Path, "lstat", return_value=info):
                with self.assertRaises(PermissionError):
                    parser._readable(path)

    def test_symlinks_are_never_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "linked.txt"
            info = os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 0, 0, 6, 0, 0, 0))
            with patch.object(Path, "lstat", return_value=info):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    parser._readable(path)

    def test_parser_child_is_killed_on_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder)
            (corpus / "slow.txt").write_text("test", encoding="utf-8")
            units, errors = parser.parse_corpus(corpus, file_timeout=0.001)
            self.assertEqual(units, [])
            self.assertIn("time limit", errors[0]["error"])

    def test_empty_corpus_and_total_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(parser.parse_corpus(Path(folder)), ([], []))
            (Path(folder) / "one.txt").write_text("text", encoding="utf-8")
            units, errors = parser.parse_corpus(Path(folder), total_timeout=0)
            self.assertEqual(units, [])
            self.assertIn("deadline", errors[0]["error"])

    def test_ocr_rows_keep_common_columns_across_blocks(self):
        header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        entries = [
            "5\t1\t1\t1\t1\t1\t100\t100\t35\t20\t95\tB11",
            "5\t1\t2\t1\t1\t1\t300\t101\t35\t20\t95\tB14",
            "5\t1\t3\t1\t1\t1\t100\t180\t35\t20\t95\tGND",
            "5\t1\t4\t1\t1\t1\t300\t180\t95\t20\t95\tTHERM_ALERT#",
        ]
        output = parser._ocr_layout(header + "\n".join(entries), 600)
        rows = output.splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].index("B11"), rows[1].index("GND"))
        self.assertEqual(rows[0].index("B14"), rows[1].index("THERM_ALERT#"))

    @unittest.skipUnless(importlib.util.find_spec("defusedxml"), "defusedxml is not installed")
    def test_docx_table_and_deleted_text(self):
        xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
               '<w:p><w:r><w:t>Current revision</w:t></w:r><w:del><w:r><w:delText>obsolete secret</w:delText></w:r></w:del></w:p>'
               '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Part</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Limit</w:t></w:r></w:p></w:tc></w:tr>'
               '<w:tr><w:tc><w:p><w:r><w:t>ABC-9</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>84</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
               '</w:body></w:document>')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "table.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", xml)
            payload = parser._worker(path, "nested/table.docx")
            text = "\n".join(u["text"] for u in payload["units"])
            self.assertEqual(payload["errors"], [])
            self.assertIn("Part: ABC-9 | Limit: 84", text)
            self.assertNotIn("obsolete", text)

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "openpyxl is not installed")
    def test_spreadsheet_uncached_formula_is_not_fabricated(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "formulas.xlsx"
            workbook = openpyxl.Workbook()
            workbook.active.append(["Product", "Limit"])
            workbook.active.append(["ABC-9", "=40+2"])
            workbook.save(path)
            payload = parser._worker(path, "formulas.xlsx")
            text = "\n".join(u["text"] for u in payload["units"])
            self.assertEqual(payload["errors"], [])
            self.assertIn("formula with no cached result: =40+2", text)

    @unittest.skipUnless(importlib.util.find_spec("fitz"), "PyMuPDF is not installed")
    def test_encrypted_pdf_skipped_and_neighbor_read(self):
        import fitz
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder)
            with fitz.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), "Maximum junction temperature: 91 degrees Celsius")
                document.save(corpus / "good.pdf")
                document.save(corpus / "locked.pdf", encryption=fitz.PDF_ENCRYPT_AES_256,
                              owner_pw="owner-only", user_pw="unavailable")
                document.save(corpus / "owner-only.pdf", encryption=fitz.PDF_ENCRYPT_AES_256,
                              owner_pw="owner-only", user_pw="")
            units, errors = parser.parse_corpus(corpus)
            self.assertEqual({u["path"] for u in units}, {"good.pdf"})
            self.assertEqual(len(errors), 2)
            self.assertTrue(all("encrypted PDF skipped" in e["error"] for e in errors))

    @unittest.skipUnless(importlib.util.find_spec("fitz"), "PyMuPDF is not installed")
    def test_mixed_pdf_page_ocr_is_used_and_text_survives_ocr_failure(self):
        import fitz
        from PIL import Image
        import io
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mixed.pdf"
            image = io.BytesIO()
            Image.new("RGB", (500, 500), "white").save(image, format="PNG")
            with fitz.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), "Detailed equipment overview with a separate scanned label below")
                page.insert_image(fitz.Rect(70, 150, 500, 650), stream=image.getvalue())
                document.save(path)
            with patch.object(parser, "_ocr", return_value="BOARD REVISION REV-F3") as ocr:
                payload = parser._worker(path, "mixed.pdf")
                self.assertTrue(ocr.called)
                self.assertIn("REV-F3", payload["units"][0]["text"])
            with patch.object(parser, "_ocr", side_effect=TimeoutError("ocr timeout")):
                payload = parser._worker(path, "mixed.pdf")
                self.assertIn("Detailed equipment overview", payload["units"][0]["text"])


if __name__ == "__main__":
    unittest.main()
