import json
import os.path
from concurrent.futures import ThreadPoolExecutor

from agentless.fl.combine import combine_file_level
from agentless.fl.localize import localize_instance, localize_irrelevant, merge
from agentless.fl.retrieve import retrieve
from agentless.repair.repair import repair
from agentless.repair.rerank import normalize_patches, _load_results, majority_voting
from agentless.test.generate_reproduction_tests import generate_tests, post_process_tests, normalize_tests, \
    test_selection
from agentless.test.run_regression_tests import _run_regression
from agentless.test.run_reproduction_tests import _run_reproduction_tests
from agentless.test.select_regression_tests import select_tests
from agentless.util.arguments import LocalizationArgs, RetrievalArgs, CombineArgs, RepairArgs, RegressionTestsArgs, \
    SelectRegressionTestsArgs, GenerateTestArgs, RunReproductionTestsArgs, RerankArgs
from tqdm import tqdm

from nemo_skills.inference.generate import GenerationTask


class AgentlessGenerationTask(GenerationTask):
    def _localize_suspicious_files(self, data_point, data, save_dir):
        args = LocalizationArgs(
            output_folder=os.path.join(save_dir, "file_level/"),
            file_level=True,
            num_threads=10,
            skip_existing=True,
        )
        localize_instance(data_point, args, data, start_file_locs=None, existing_instance_ids=set())

    def _remove_irrelevant_folders(self, save_dir):
        args = LocalizationArgs(
            output_folder=os.path.join(save_dir, "file_level_irrelevant/"),
            file_level=True,
            irrelevant=True,
            num_threads=10,
            skip_existing=True,
        )
        localize_irrelevant(args)

    def _retrieve_from_relevant_folders(self, save_dir):
        args = RetrievalArgs(
            index_type="simple",
            filter_type="given_files",
            filter_file=os.path.join(save_dir, "file_level_irrelevant/loc_outputs.jsonl"),
            output_folder=os.path.join(save_dir, "retrieval_embedding/"),
            persist_dir=os.path.join(save_dir, "embedding/"),
            num_threads=10,
        )
        retrieve(args)

    def _merge_localizations(self, save_dir):
        args = CombineArgs(
            retrieval_loc_file=os.path.join(save_dir, "retrieval_embedding/retrieve_locs.jsonl"),
            model_loc_file=os.path.join(save_dir, "file_level/loc_outputs.jsonl"),
            top_n=3,
            output_folder=os.path.join(save_dir, "file_level_combined/")
        )
        combine_file_level(args)

    def _find_related_elements(self, data_point, data, save_dir):
        args = LocalizationArgs(
            related_level=True,
            output_folder=os.path.join(save_dir, "related_elements/"),
            top_n=3,
            compress_assign=True,
            compress=True,
            start_file=os.path.join(save_dir, "file_level_combined/combined_locs.jsonl"),
            num_threads=10,
            skip_existing=True
        )
        localize_instance(data_point, args, data, start_file_locs=None, existing_instance_ids=set())

    def _localize_to_edit_locations(self, data_point, data, save_dir):
        args = LocalizationArgs(
            fine_grain_line_level=True,
            output_folder=os.path.join(save_dir, "edit_location_samples/"),
            top_n=3,
            compress=True,
            temperature=0.8,
            num_samples=4,
            start_file=os.path.join(save_dir, "related_elements/loc_outputs.jsonl"),
            num_threads=10,
            skip_existing=True
        )
        localize_instance(data_point, args, data, start_file_locs=None, existing_instance_ids=set())

    def _separate_edit_locations(self, save_dir):
        args = LocalizationArgs(
            merge=True,
            output_folder=os.path.join(save_dir, "edit_location_individual/"),
            top_n=3,
            num_samples=4,
            start_file=os.path.join(save_dir, "edit_location_samples/loc_outputs.jsonl"),
        )
        merge(args)

    def _generate_patches(self, save_dir, num_repair_samples):
        for i in range(num_repair_samples + 1):
            args = RepairArgs(
                loc_file=os.path.join(save_dir, f"edit_location_individual/loc_merged_{i}-{i}_outputs.jsonl"),
                output_folder=os.path.join(save_dir, f"repair_sample_{i + 1}/"),
                loc_interval=True,
                top_n=3,
                context_window=10,
                max_samples=10,
                cot=True,
                diff_format=True,
                gen_and_process=True,
                num_threads=2,
            )
            repair(args)

    def _prepare_regression_tests(self, save_dir):
        run_args = RegressionTestsArgs(
            run_id="generate_regression_tests",
            output_file=os.path.join(save_dir, "passing_tests.jsonl")
        )
        _run_regression(run_args)

        select_args = SelectRegressionTestsArgs(
            passing_tests=run_args.output_file,
            output_folder=os.path.join(save_dir, "select_regression/"),
        )
        select_tests(select_args)

    def _run_regression_on_patches(self, save_dir, num_repair_samples):
        for i in range(1, num_repair_samples + 2):
            folder_name = f"repair_sample_{i}"
            def run_single(num):
                args = RegressionTestsArgs(
                    regression_tests=os.path.join(save_dir, "select_regression/output.jsonl"),
                    predictions_path=os.path.join(save_dir, folder_name, f"output_{num}_processed.jsonl"),
                    run_id=f"{folder_name}_regression_{num}",
                    num_workers=10
                )
                _run_regression(args)
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(run_single, range(10))

    def _generate_reproduction_tests(self, save_dir):
        args = GenerateTestArgs(
            max_samples=40,
            output_folder=os.path.join(save_dir, "reproduction_test_samples/"),
            num_threads=10,
        )
        args.output_file = os.path.join(args.output_folder, "output.jsonl")
        generate_tests(args)
        args.raw_output_file = args.output_file
        for i in range(args.max_samples):
            args.output_file = args.raw_output_file.replace(".jsonl", f"_{i}_processed_reproduction_test.jsonl")
            args.select_id = i
            post_process_tests(args)

    def _run_and_filter_reproduction_tests(self, save_dir):
        def run_single(num):
            args = RunReproductionTestsArgs(
                run_id=f"reproduction_test_generation_filter_sample_{num}",
                test_jsonl=os.path.join(save_dir, f"reproduction_test_samples/output_{num}_processed_reproduction_test.jsonl"),
                num_workers=6,
                testing=True
            )
            _run_reproduction_tests(args)
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(run_single, range(0, 40))

    def _select_final_reproduction_test(self, save_dir):
        args = GenerateTestArgs(
            max_samples=40,
            output_folder=os.path.join(save_dir, "reproduction_test_samples/"),
            output_file="reproduction_tests.jsonl",
            select=True,
        )
        normalize_tests(args)
        test_selection(args)

    def _evaluate_patches_with_repro_tests(self, save_dir, num_repair_samples):
        for i in range(num_repair_samples + 2):
            run_id_prefix = f"repair_sample_{i}"
            def run_single(num):
                args = RunReproductionTestsArgs(
                    test_jsonl="results/swe-bench-lite/reproduction_test_samples/reproduction_tests.jsonl",
                    predictions_path=os.path.join(save_dir, f"{run_id_prefix}/output_{num}_processed.jsonl"),
                    run_id=f"{run_id_prefix}_reproduction_{num}",
                    num_workers=10
                )
                _run_reproduction_tests(args)
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(run_single, range(10))

    def _rerank_and_select_final_patch(self, save_dir, num_repair_samples):
        patch_folders = ','.join([os.path.join(save_dir, f"repair_sample_{i}") for i in range(1, num_repair_samples + 2)])
        args = RerankArgs(
            patch_folder=patch_folders,
            num_samples=40,
            deduplicate=True,
            regression=True,
            reproduction=True
        )
        normalize_patches(args)
        _load_results(args)
        majority_voting(args)

    def sync_loop(self, data):
        save_dir = os.path.dirname(self.cfg.output_file)
        with open(self.cfg.output_file, "at", encoding="utf-8", buffering=1) as fout:
            for data_point in tqdm(data, desc="Remaining generations"):
                num_additional_repair_samples = 3

                # 1. Localize suspicious files using LLM
                self._localize_suspicious_files(data_point, data, save_dir)

                # 2. Remove irrelevant folders before running embedding-based retrieval localization.
                self._remove_irrelevant_folders(save_dir)

                # 3. Retrieval from relevant folders, filtering out irrelevant files.
                self._retrieve_from_relevant_folders(save_dir)

                # 4. Merge LLM-predicted suspicious files with embedding-based retrieval, create final releveant files.
                self._merge_localizations(save_dir)

                # 5. Find related elements in suspicious files.
                self._find_related_elements(data_point, data, save_dir)

                # 6. Localize to edit locations using related elements.
                self._localize_to_edit_locations(data_point, data, save_dir)

                # 7. Separate individual sets of edit locations.
                self._separate_edit_locations(save_dir)

                # 8. Generate patches using the LLM for repairing.
                self._generate_patches(save_dir, num_additional_repair_samples)

                # 9. Select regression tests (already exist in repo) to run. Select passing tests.
                # 10. Ask the LLM to remove any tests that should not be ran.
                self._prepare_regression_tests(save_dir)

                # 11. Run the selected tests on the generated repair patches.
                self._run_regression_on_patches(save_dir, num_additional_repair_samples)

                # 12. Generate reproduction tests to see if it solves the original issues using LLM.
                self._generate_reproduction_tests(save_dir)

                # 13. Run reproduction tests to see if they can reproduce the issue, and filter those that do not.
                self._run_and_filter_reproduction_tests(save_dir)

                # 14. Apply majority voting to select one reproduction test per issue.
                self._select_final_reproduction_test(save_dir)

                # 15. Evaluate generated patches using selected reproduction test.
                self._evaluate_patches_with_repro_tests(save_dir, num_additional_repair_samples)

                # 16. Perform re-ranking using the regression/reproduction test results to select final patch.
                self._rerank_and_select_final_patch(save_dir, num_additional_repair_samples)

                fout.write(json.dumps({**data_point, "completed": True}) + '\n')
