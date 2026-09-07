"""Retrieval must work where the gateway does not.

WHAT THIS PROTECTS

    The deployed app returned available:False with
    "ModuleNotFoundError: No module named 'second_brain'" and answered every
    question ungrounded. It degraded silently - which is correct behaviour
    and exactly why nobody noticed - while the shelf grew to 64 books the app
    could not quote.

    So the load-bearing test is the one that blocks second_brain the way the
    container does and asserts grounding is STILL available.
"""
import builtins
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import local_corpus as lc


class TestTheBundleShips(unittest.TestCase):
    def test_the_bundle_is_present_in_the_repo(self):
        """Nixpacks builds from the repo and has no access to the corpus on
        the author's machine, so the bundle has to be committed."""
        self.assertTrue(lc.BUNDLE.exists(), f"missing {lc.BUNDLE}")

    def test_the_bundle_is_small_enough_to_commit(self):
        mb = lc.BUNDLE.stat().st_size / 1e6
        self.assertLess(mb, 40, f"{mb:.1f} MB is too large to keep in git")

    def test_it_is_tracked_by_git_not_ignored(self):
        r = subprocess.run(["git", "check-ignore", str(lc.BUNDLE)],
                           cwd=ROOT, capture_output=True)
        self.assertNotEqual(r.returncode, 0,
                            "the bundle is gitignored, so it will not deploy")


class TestSearch(unittest.TestCase):
    def test_it_returns_hits_with_sources(self):
        hits = lc.search("de Broglie wavelength of an electron", top_k=5)
        self.assertTrue(hits)
        for h in hits:
            self.assertTrue(h["source"])
            self.assertTrue(h["text"])

    def test_scores_are_positive_and_better_is_higher(self):
        """The gateway flips bm25's sign so callers can treat it like every
        other relevance number. This must match, or tutor.py's MIN_RAW_SCORE
        filter means something different on each path."""
        hits = lc.search("quantum harmonic oscillator ladder operators", top_k=5)
        scores = [h["raw_score"] for h in hits]
        self.assertTrue(all(s > 0 for s in scores), scores)
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_clear_the_tutors_threshold(self):
        import tutor
        hits = lc.search("Schrodinger equation for the hydrogen atom", top_k=3)
        self.assertGreater(hits[0]["raw_score"], tutor.MIN_RAW_SCORE)

    def test_a_multi_word_question_is_ORed_not_ANDed(self):
        """ANDed, a long question matches nothing. Measured before the fix:
        'renormalisation field theory' returned zero hits against a corpus
        holding four QFT texts."""
        self.assertIn(" OR ", lc.to_match_query("renormalisation field theory"))
        self.assertTrue(lc.search("renormalisation field theory", top_k=3))

    def test_nonsense_returns_nothing_rather_than_noise(self):
        self.assertEqual(lc.search("zzqqxx wrrgle", top_k=3), [])

    def test_an_empty_question_is_not_a_query(self):
        self.assertEqual(lc.search("   ", top_k=3), [])

    def test_hits_carry_page_numbers(self):
        """The re-ingest added page metadata; citations can be page-accurate."""
        hits = lc.search("cavity decoherence superposition", top_k=6)
        self.assertTrue(any(h.get("page_start") for h in hits))

    def test_solutions_manuals_are_not_in_the_bundle(self):
        """A worked answer key ranks well on anything phrased like a problem,
        and then the tutor explains from the answer rather than the physics."""
        hits = lc.search("infinite square well energy eigenvalues", top_k=10)
        bad = [h["source"] for h in hits if "solution manual" in h["source"].lower()]
        self.assertEqual(bad, [])

    def test_the_newly_indexed_books_are_reachable(self):
        hits = lc.search("Rydberg atom cavity photon decoherence experiment",
                         top_k=10)
        sources = " ".join(os.path.basename(h["source"]) for h in hits)
        self.assertIn("Exploring the quantum", sources)


class TestItReportsWhatItIs(unittest.TestCase):
    def test_status_names_the_corpus_size(self):
        lc.search("hydrogen atom", top_k=1)          # force the build
        st = lc.status()
        self.assertTrue(st["built"])
        self.assertGreater(st["chunks"], 10_000)
        self.assertGreater(st["books"], 40)


class TestTheTutorFallsBackToIt(unittest.TestCase):
    """The whole point: with second_brain unimportable, as in the container,
    grounding must still be available."""

    def test_grounding_is_available_without_the_gateway(self):
        code = (
            "import builtins,sys; sys.path.insert(0,%r)\n"
            "_r=builtins.__import__\n"
            "def b(n,*a,**k):\n"
            "    if n.startswith('second_brain'):\n"
            "        raise ModuleNotFoundError(\"No module named 'second_brain'\")\n"
            "    return _r(n,*a,**k)\n"
            "builtins.__import__=b\n"
            "import tutor\n"
            "r=tutor.retrieve_evidence('decoherence of a cavity superposition')\n"
            "print(r['available'], len(r.get('kept',[])), "
            "r['kept'][0]['corpus'] if r.get('kept') else None)\n"
        ) % str(ROOT)
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        line = out.stdout.strip().splitlines()[-1]
        self.assertTrue(line.startswith("True"), line)
        self.assertIn("quantum-books", line)


if __name__ == "__main__":
    unittest.main()


class TestCitationsCarryPages(unittest.TestCase):
    """The re-ingest put page numbers on 164,219 of 165,423 chunks where none
    existed before. tutor.py was dropping them when it normalised hits, so a
    citation could name the book but never the page - the difference between
    "Haroche says so" and a claim the reader can go and check."""

    def test_the_tutor_keeps_the_page_on_each_hit(self):
        code = (
            "import builtins,sys; sys.path.insert(0,%r)\n"
            "_r=builtins.__import__\n"
            "def b(n,*a,**k):\n"
            "    if n.startswith('second_brain'):\n"
            "        raise ModuleNotFoundError('no gateway')\n"
            "    return _r(n,*a,**k)\n"
            "builtins.__import__=b\n"
            "import tutor\n"
            "r=tutor.retrieve_evidence('decoherence of a cavity superposition')\n"
            "print(sum(1 for h in r['kept'] if h.get('page')), len(r['kept']))\n"
        ) % str(ROOT)
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        withpage, total = map(int, out.stdout.strip().splitlines()[-1].split())
        self.assertGreater(total, 0)
        self.assertGreater(withpage, 0, "no hit carried a page number")

    def test_the_source_block_shown_to_the_model_names_the_page(self):
        src = (ROOT / "tutor.py").read_text()
        i = src.index("raw relevance")
        self.assertIn("p.", src[max(0, i - 200):i],
                      "the prompt's source block does not cite a page")


class TestRanking(unittest.TestCase):
    """The re-rank constants were measured, so the tests assert the measured
    outcome rather than the mechanism. A change that moves these numbers down
    is a regression whatever it does to the code."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tests"))
        import retrieval_eval as ev
        cls.ev = ev
        cls.r = ev.evaluate(lc.search)

    def test_hit_at_3_beats_the_recorded_bm25_baseline(self):
        """BM25 alone scored 68% on this set."""
        self.assertGreaterEqual(self.r["hit3_pct"], 0.71)

    def test_hit_at_5_beats_the_recorded_baseline(self):
        """BM25 alone scored 75%."""
        self.assertGreaterEqual(self.r["hit5_pct"], 0.82)

    def test_the_ranking_still_returns_several_books(self):
        """A re-rank that collapses onto one book has given the reader one
        opinion, not evidence."""
        self.assertGreater(self.r["mean_spread"], 2.5)

    def test_no_popular_level_book_answers_a_technical_question(self):
        """Two did before the level prior: 'The Joy of Quantum Computing' on
        the density matrix, and 'Why Nobody Understands Quantum Physics' on
        the WKB approximation."""
        pop = ("why nobody understands", "joy of quantum computing",
               "simply quantum physics")
        offenders = []
        for question, _expected in self.ev.CASES:
            for h in lc.search(question, top_k=3):
                name = os.path.basename(h["source"]).lower()
                if any(p in name for p in pop):
                    offenders.append((question[:40], name[:40]))
        self.assertEqual(offenders, [])

    def test_the_level_map_shipped_with_the_bundle(self):
        lc.search("hydrogen atom", top_k=1)          # force the build
        self.assertGreater(len(lc._levels), 50,
                           "the curated levels did not ship in the bundle")

    def test_a_book_with_its_subject_in_the_title_is_reachable(self):
        """The title affinity exists for questions like this one."""
        hits = lc.search("How did von Neumann formalise the measurement process?",
                         top_k=5)
        names = " ".join(os.path.basename(h["source"]) for h in hits).lower()
        self.assertIn("neumann", names)

    def test_rejected_hypothesis_is_documented_not_silently_dropped(self):
        """Corpus-specific stopwording was tried and measured worse. The next
        person to have the same idea should find that out from the code."""
        src = (ROOT / "local_corpus.py").read_text()
        self.assertIn("REJECTED", src)
