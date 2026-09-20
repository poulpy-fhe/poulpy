#!/usr/bin/env python3
"""Regression checks for inventory drift and required executable coverage."""
import copy
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import check_core_contracts as checker

SOURCE = "poulpy-core/src/oep/example.rs"
REFERENCE = "poulpy-core/src/reference/example.rs"
TEST = "poulpy-core/src/test_suite/parity/example.rs"


def function(name, source, has_body=True):
    return {"crate_id": 0, "name": name, "span": {"filename": source},
            "inner": {"function": {"has_body": has_body}}}


class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.metadata = {"index": {
            "1": {"crate_id": 0, "name": "ExampleImpl", "span": {"filename": SOURCE},
                  "inner": {"trait": {"items": [2]}}},
            "2": function("apply", SOURCE, False),
            "3": function("apply_reference", REFERENCE),
            "4": function("test_example", TEST),
        }}
        self.manifest = {
            "methods": {"ExampleImpl::apply": {"source": SOURCE, "dispatch": [SOURCE + "#apply"],
                        "reference": [REFERENCE + "#apply_reference"], "tests": ["example"]}},
            "exceptions": {},
            "tests": {"example": {"function": TEST + "#test_example", "registration": "example", "oracle": "exact reference"}},
            "backends": {"ExampleBackend": {"package": "poulpy-cpu-example", "test_prefix": "tests::core_parity_example::", "reference_backend": "FFT64Ref"}},
        }
        self.registrations = {"ExampleBackend": ("tests::core_parity_example::", {"example": "test_example"}, "poulpy_cpu_ref::FFT64Ref")}

    def check(self, log=None):
        with patch.object(checker, "registrations", return_value=self.registrations), \
             patch.object(checker, "encryption_registrations", return_value=({}, {})):
            if log is None:
                return checker.check(self.manifest, self.metadata, [])
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "tests.log"
                path.write_text(log)
                return checker.check(self.manifest, self.metadata, [("ExampleBackend", path)])

    def test_complete_mapping_and_executed_test_pass(self):
        self.assertEqual(self.check("test tests::core_parity_example::example ... ok\n"), [])

    def test_new_unmapped_method_is_rejected(self):
        self.metadata["index"]["1"]["inner"]["trait"]["items"].append(5)
        self.metadata["index"]["5"] = function("assign", SOURCE, False)
        self.assertIn("missing method: ExampleImpl::assign", self.check())

    def test_deleted_method_is_not_silently_kept(self):
        self.manifest["methods"]["ExampleImpl::removed"] = copy.deepcopy(self.manifest["methods"]["ExampleImpl::apply"])
        self.assertIn("stale method: ExampleImpl::removed", self.check())

    def test_abstract_declaration_cannot_claim_reference_body(self):
        self.metadata["index"]["3"]["inner"]["function"]["has_body"] = False
        self.assertTrue(any("missing reference body" in error for error in self.check()))

    def test_missing_supported_registration_is_rejected(self):
        self.registrations["ExampleBackend"][1].clear()
        self.assertIn("missing supported registration: ExampleBackend/example", self.check())

    def test_removed_backend_suite_is_rejected(self):
        self.registrations.clear()
        self.assertIn("backend contract suite not registered: ExampleBackend", self.check())

    def test_new_backend_must_be_inventoried(self):
        self.registrations["NewBackend"] = ("tests::new::", {"example": "test_example"}, "poulpy_cpu_ref::FFT64Ref")
        self.assertIn("backend missing from contract inventory: NewBackend", self.check())

    def test_ignored_failed_filtered_and_listed_tests_do_not_count(self):
        for log in ["", "tests::core_parity_example::example: test\n",
                    "test tests::core_parity_example::example ... ignored\n",
                    "test tests::core_parity_example::example ... FAILED\n"]:
            with self.subTest(log=log):
                self.assertTrue(any("contract test did not pass" in error for error in self.check(log)))

    def test_another_backends_success_does_not_count(self):
        self.assertTrue(any("contract test did not pass" in error for error in
                            self.check("test tests::core_parity_other::example ... ok\n")))

    def test_accelerated_reference_cannot_replace_portable_oracle(self):
        self.registrations["ExampleBackend"] = ("tests::core_parity_example::", {"example": "test_example"}, "FFT64Avx")
        self.assertIn("incorrect portable reference backend: ExampleBackend", self.check())

    def test_oracle_name_in_another_crate_does_not_count(self):
        self.registrations["ExampleBackend"] = ("tests::core_parity_example::", {"example": "test_example"}, "crate::FFT64Ref")
        self.assertIn("incorrect portable reference backend: ExampleBackend", self.check())

    def test_wrong_registered_prefix_is_rejected(self):
        self.manifest["backends"]["ExampleBackend"]["test_prefix"] = "tests::wrong::"
        self.assertIn("incorrect test prefix: ExampleBackend", self.check())


if __name__ == "__main__":
    unittest.main()
