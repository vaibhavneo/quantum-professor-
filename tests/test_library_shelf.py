"""The curated shelf: integrity of the book catalogue and its serialization.

WHY THESE EXIST

    The shelf grew from 13 books to 64 in one change. At that size the
    failure mode is not "a book is missing" but "a book points at a topic
    that does not exist", "two books share an id and one silently replaces
    the other", or "a level string is misspelled so the book never appears
    at any level". None of those raise; they just quietly remove a book from
    the app.

    The authors test pins a real defect: `Book.authors` is a str, and
    serialize.py called list() on it, splatting every name into single
    characters. web/app.js renders `authors.join(", ")`, so every book card
    read "D, a, v, i, d,  , J, .". It shipped that way for all 13 books.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library import BOOKS, LEVELS, TOPICS, books_at_level
from serialize import book_to_dict


class TestShelfIntegrity(unittest.TestCase):
    def test_every_book_topic_is_a_real_topic(self):
        """A book pointing at a non-existent topic is invisible to any
        topic-driven recommendation, and nothing raises."""
        bad = [(b.id, t) for b in BOOKS.values()
               for t in b.topics if t not in TOPICS]
        self.assertEqual(bad, [])

    def test_every_book_level_is_a_real_level(self):
        bad = [(b.id, b.level) for b in BOOKS.values() if b.level not in LEVELS]
        self.assertEqual(bad, [])

    def test_book_ids_are_unique(self):
        """BOOKS is built as a dict comprehension keyed on id, so a duplicate
        id does not raise - the second entry silently replaces the first."""
        import re
        src = (Path(__file__).resolve().parents[1] / "library.py").read_text()
        ids = re.findall(r'^\s+id="([a-z0-9\-]+)",', src, re.M)
        dupes = {i for i in ids if ids.count(i) > 1}
        self.assertEqual(dupes, set())

    def test_every_book_carries_a_reason_to_read_it(self):
        thin = [b.id for b in BOOKS.values() if len(b.why_read.strip()) < 40]
        self.assertEqual(thin, [], "a shelf entry with no argument for "
                                   "reading it is a listing, not a "
                                   "recommendation")

    def test_every_level_has_books(self):
        for lv in LEVELS:
            with self.subTest(level=lv):
                self.assertTrue(books_at_level(lv), f"no books at {lv}")

    def test_the_shelf_covers_the_advanced_topics(self):
        """The topics most likely to be asked about after a first course had
        one or two books between them before the expansion."""
        for topic in ("quantum-optics", "qft-fundamentals", "quantum-information",
                      "path-integrals", "decoherence", "second-quantization"):
            with self.subTest(topic=topic):
                covering = [b.id for b in BOOKS.values() if topic in b.topics]
                self.assertGreaterEqual(len(covering), 2, topic)


class TestAuthorsAreNames(unittest.TestCase):
    """`list()` on a str splats it into characters."""

    def test_authors_serialize_as_names_not_characters(self):
        for b in BOOKS.values():
            authors = book_to_dict(b)["authors"]
            with self.subTest(book=b.id):
                self.assertTrue(all(len(a) > 1 for a in authors),
                                f"{b.id}: {authors[:6]}")

    def test_a_multi_author_book_splits_on_the_separator(self):
        got = book_to_dict(BOOKS["feynman-vol3"])["authors"]
        self.assertEqual(got, ["Richard P. Feynman", "Robert B. Leighton",
                               "Matthew Sands"])

    def test_a_single_author_book_is_one_entry(self):
        self.assertEqual(book_to_dict(BOOKS["griffiths"])["authors"],
                         ["David J. Griffiths"])

    def test_no_author_entry_is_blank(self):
        for b in BOOKS.values():
            for a in book_to_dict(b)["authors"]:
                self.assertTrue(a.strip(), b.id)


if __name__ == "__main__":
    unittest.main()
