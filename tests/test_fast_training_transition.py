from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from native_core.env import NativeRoyaleEnv, NativeHostError
from native_core.client import IDEMPOTENT_OPS


class _WireEnv(NativeRoyaleEnv):
    def _request(self, payload):
        self.sent = payload
        return {"result": self.reply}

    @staticmethod
    def _enrich_episode(episode):
        return dict(episode)

    def _enrich_training_state(self, state):
        return dict(state)


class FastTransitionContractTests(unittest.TestCase):
    def test_fast_mode_preserves_decoded_contract_and_is_not_replayed(self):
        results = []
        for mode in ("legacy", "fast-json"):
            env = _WireEnv(transition_mode=mode)
            env.reply = {"joint_action": {"actions": []},
                         "episode": {"terminated": False, "truncated": False},
                         "state": {"tick": 105, "entities": [{"hp": 55}]}}
            results.append(env.joint_training_transition([], steps=5))
            self.assertEqual(env.sent["steps"], 5)
            expected = "joint_training_transition_fast_v1" if mode == "fast-json" else "joint_training_transition_v1"
            self.assertEqual(env.sent["op"], expected)
            self.assertNotIn(expected, IDEMPOTENT_OPS)
        self.assertEqual(results[0], results[1])

    def test_invalid_mode_or_profile_combination_rejected_before_actions(self):
        with self.assertRaises(ValueError):
            _WireEnv(transition_mode="other")
        with self.assertRaises(ValueError):
            _WireEnv(transition_mode="fast-json", profile_native=True)

    def test_missing_nonterminal_state_is_not_a_valid_empty_battle(self):
        env = _WireEnv(transition_mode="fast-json")
        env.reply = {"joint_action": {"actions": []}, "episode": {"terminated": False}}
        with self.assertRaises(NativeHostError):
            env.joint_training_transition([], steps=5)
        env.reply["episode"]["terminated"] = True
        self.assertNotIn("state", env.joint_training_transition([], steps=5))


class NativeFragmentJavaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        jdk = os.environ.get("CR_SANDBOX_JDK")
        cls.javac = str(Path(jdk)/"bin/javac.exe") if jdk else shutil.which("javac")
        cls.java = str(Path(jdk)/"bin/java.exe") if jdk else shutil.which("java")
        if not cls.java or not cls.javac or not Path(cls.javac).is_file():
            raise unittest.SkipTest("set CR_SANDBOX_JDK to test the Java codec")
        cls.temp = tempfile.TemporaryDirectory()
        cls.classes = cls.temp.name
        subprocess.run([cls.javac,"--release","8","-d",cls.classes,
                        str(root/"android_probe/java/royale/nativehost/TrainingTransitionResponse.java"),
                        str(root/"tests/java/TrainingTransitionResponseHarness.java")],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def encode(self, *values):
        result = subprocess.run([self.java,"-Dfile.encoding=UTF-8","-cp",self.classes,
                                 "TrainingTransitionResponseHarness",*values],
                                capture_output=True,text=True,encoding="utf-8")
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def test_nested_unicode_and_escaped_strings(self):
        a = {"actions":[{"accepted":True}]}
        e = {"terminated":False,"crowns":[0,0]}
        s = {"tick":105,"name":"雪人 {x} \"text\"\nnext","entities":[{"hp":55,"known":True}]}
        result = self.encode(*(json.dumps(x,ensure_ascii=True) for x in (a,e,s)))
        self.assertEqual(result["result"],{"joint_action":a,"episode":e,"state":s})
        self.assertEqual(result["op"],"joint_training_transition_fast_v1")

    def test_terminal_omits_state(self):
        result = self.encode('{"actions":[]}','{"terminated":true}')
        self.assertNotIn("state",result["result"])


if __name__ == "__main__":
    unittest.main()
