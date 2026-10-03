"""Experiment conditions, CLI overrides, and recorded run provenance; no game workers."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from agents.nn import experiment

try:
    from agents.nn import train
except ImportError:
    train = None


@unittest.skipIf(train is None, "requires nn dependencies")
class ExperimentTests(unittest.TestCase):
    def resolve(self, *argv):
        return experiment.resolve(train.parser(), list(argv))

    def test_presets_resolve_and_cli_overrides_win(self):
        for name in ("insatiable", "starter-hp", "starter-turn-cost", "library-act1"):
            args, meta = self.resolve('--experiment', name, '--name', 'trial', '--device', 'cpu',
                                      '--hours', '0.1', '--no-bucket', '--init-from', '/tmp/source.pt',
                                      '--pool-file', '/tmp/library.json')
            self.assertEqual(args.hours, 0.1)
            self.assertFalse(args.bucket)
            self.assertEqual(meta['resolved']['device'], 'cpu')
            self.assertEqual(meta['definition']['id'], name)
        self.assertEqual(self.resolve('--experiment', 'starter-hp', '--name', 'trial')[0].turn_cost_hp, 0)

    def test_required_inputs_and_resume(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.resolve('--experiment', 'library-act1', '--name', 'trial')
        args, _ = self.resolve('--experiment', 'library-act1', '--name', 'trial', '--resume',
                               '--pool-file', '/tmp/library.json')
        self.assertIsNone(args.init_from)
        self.assertFalse(args.extend_vocab)

    def test_resume_does_not_hide_conflicting_explicit_initialization(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.main(['--experiment', 'starter-hp', '--name', 'trial', '--resume',
                        '--init-from', '/tmp/checkpoint.pt', '--dry-run'])

    def test_misspelled_conditions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'experiment.json'
            path.write_text(json.dumps(dict(format_version=1, id='typo', hypothesis='x', training=dict(rewrd='hp'))))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.resolve('--experiment', str(path), '--name', 'trial')

    def test_dry_run_never_builds_a_pool_or_writes_a_run(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'trial'
            with patch.object(train.pools, 'pool', side_effect=AssertionError('started game')), contextlib.redirect_stdout(io.StringIO()) as output:
                train.main(['--experiment', 'insatiable', '--name', str(out), '--dry-run'])
            report = json.loads(output.getvalue())
            self.assertEqual(report['selection_metric'], 'win')
            self.assertEqual(report['resolved']['eval_seeds'], 100)
            self.assertFalse(out.exists())

    def test_cannot_relabel_an_ad_hoc_run_as_an_experiment_on_resume(self):
        _, meta = self.resolve('--experiment', 'starter-hp', '--name', 'trial')
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            experiment.record(Path(tmp), meta, True)

    def test_resume_preserves_original_artifact_and_rejects_changed_definition(self):
        _, meta = self.resolve('--experiment', 'starter-hp', '--name', 'trial')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            experiment.record(out, meta, False)
            before = (out / 'experiment.json').read_bytes()
            self.assertEqual(experiment.record(out, None, True), meta)
            with self.assertRaises(ValueError):
                experiment.record(out, dict(meta, sha256='changed'), True)
            self.assertEqual((out / 'experiment.json').read_bytes(), before)
