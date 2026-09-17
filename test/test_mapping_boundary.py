# Verify backend ownership with a tiny matrix search
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from voyager_compiler.mapping.search import prepare_search
from voyager_compiler.mapping.operations import parse_operation
from voyager_compiler.mapping.driver import generate_tilings
from voyager_compiler.mapping.results import write_tilings
from test_cim_timing import small_target
from voyager_compiler.mapping.target import SATarget
from test_mapping_streams import small_model, small_epilogues


# Supply an eight-lane SA target without a synthesized accelerator
def small_sa():
    return SATarget(k=8, n=8, input_buffer_words=256, weight_buffer_words=256,
                    accum_buffer_words=32, double_buffered_accum=False,
                    ic_port_bits=64, oc_port_bits=64, input_bits=8, weight_bits=8,
                    accum_bits=24, datatype="INT8", vector_config=dict(lanes=8),
                    output_storage=dict(matrix_results=24, accumulation_metadata=0,
                                        accumulation_writeback=8, matrix_output=64, vector_pipeline=0))


# Prepare a model from the same operation conversion used by the driver
def prepare(target, operation, options, *, epilogue=None):
    workload, vector_timing = parse_operation(target, operation, epilogue)
    return prepare_search(target, workload, vector_timing=vector_timing, options=options)

# Keep energy and winner serialization outside generic search calculations
class MappingBoundaryTests(unittest.TestCase):
    # Both backends publish results without an invented energy cost
    def test_results_without_energy_costs(self):
        for target in (small_target(), small_sa()):
            with self.subTest(backend=target.backend), TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                with patch("interstellar.cost_model.get_cost", side_effect=AssertionError("generic energy")):
                    write_tilings(*generate_tilings(small_model(), target, epilogues=small_epilogues()), output)
                report = json.loads((output / "mapping-evaluations.json").read_text())
                evaluation = report["operations"][0]["evaluation"]
                self.assertIsNone(evaluation.get("cost"))
                self.assertIsNone(evaluation.get("energy"))
                self.assertNotEqual(evaluation.get("cost_unit"), "pJ")
                self.assertEqual({path.name for path in output.iterdir()},
                                 {"tilings.txtpb", "mapping-evaluations.json", "mapping-report.md"})
