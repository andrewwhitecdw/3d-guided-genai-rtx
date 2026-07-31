"""Regression tests for ComfyUI workflow template files."""

import json
import unittest
from pathlib import Path

WORKFLOW = (Path(__file__).parent.parent / "example_workflows" / "LTX2.3_FF_LF_rtx_video.json").resolve()


def _accepts(slot_type, link_type):
    if slot_type is None or link_type is None:
        return slot_type == link_type
    return link_type in {t.strip() for t in slot_type.split(",")}


class TestWorkflowTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(WORKFLOW, encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_required_top_level_keys(self):
        for key in ("id", "revision", "nodes", "links", "definitions", "version"):
            self.assertIn(key, self.data)

    def test_all_links_reference_existing_top_level_nodes(self):
        node_ids = {node["id"] for node in self.data["nodes"]}
        for link in self.data["links"]:
            self.assertIn(link[1], node_ids, f"link {link[0]} origin node missing")
            self.assertIn(link[3], node_ids, f"link {link[0]} target node missing")

    def test_link_types_match_node_slots(self):
        outputs = {}
        inputs = {}
        for node in self.data["nodes"]:
            for slot, output in enumerate(node.get("outputs", [])):
                outputs[(node["id"], slot)] = output.get("type")
            for slot, inp in enumerate(node.get("inputs", [])):
                inputs[(node["id"], slot)] = inp.get("type")
        for link in self.data["links"]:
            lid, origin, oslot, target, tslot, ltype = link
            self.assertTrue(
                _accepts(outputs.get((origin, oslot)), ltype),
                f"link {lid} origin ({origin},{oslot}) does not accept {ltype}",
            )
            self.assertTrue(
                _accepts(inputs.get((target, tslot)), ltype),
                f"link {lid} target ({target},{tslot}) does not accept {ltype}",
            )

    def test_unique_link_ids(self):
        link_ids = [link[0] for link in self.data["links"]]
        self.assertEqual(len(link_ids), len(set(link_ids)))

    def test_last_ids_cover_used_ids(self):
        max_node_id = max(node["id"] for node in self.data["nodes"])
        max_link_id = max(link[0] for link in self.data["links"])
        self.assertGreaterEqual(self.data["last_node_id"], max_node_id)
        self.assertGreaterEqual(self.data["last_link_id"], max_link_id)

    def test_subgraph_node_interface_matches_definition(self):
        subgraphs = {sg["id"]: sg for sg in self.data.get("definitions", {}).get("subgraphs", [])}
        for node in self.data["nodes"]:
            if node["type"] not in subgraphs:
                continue
            sg = subgraphs[node["type"]]
            sg_inputs = sg.get("inputs", [])
            sg_outputs = sg.get("outputs", [])
            self.assertEqual(len(node.get("inputs", [])), len(sg_inputs))
            self.assertEqual(len(node.get("outputs", [])), len(sg_outputs))
            for i, inp in enumerate(node["inputs"]):
                self.assertEqual(inp.get("type"), sg_inputs[i]["type"])
            for i, out in enumerate(node["outputs"]):
                self.assertEqual(out.get("type"), sg_outputs[i]["type"])

    def test_required_model_references_present(self):
        expected = {
            "CheckpointLoaderSimple": {"ltx-2.3-22b-distilled-fp8.safetensors"},
            "LTXVAudioVAELoader": {"ltx-2.3-22b-distilled-fp8.safetensors"},
            "LTXAVTextEncoderLoader": {
                "gemma_3_12B_it_fp4_mixed.safetensors",
                "ltx-2.3-22b-distilled-fp8.safetensors",
            },
        }
        nodes_by_type = {}
        for node in self.data["nodes"]:
            nodes_by_type.setdefault(node["type"], []).append(node)
        for node_type, values in expected.items():
            self.assertIn(node_type, nodes_by_type, f"missing {node_type} node")
            widgets = set(nodes_by_type[node_type][0].get("widgets_values", []))
            missing = values - widgets
            self.assertFalse(missing, f"{node_type} missing widget values: {missing}")


if __name__ == "__main__":
