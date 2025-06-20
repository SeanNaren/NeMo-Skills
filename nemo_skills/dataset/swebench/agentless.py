import json
import logging
import os.path
import os.path
import shutil
from concurrent.futures import ThreadPoolExecutor

import hydra
from agentless.fl.combine import combine_file_level
from agentless.fl.localize import localize_instance, merge, localize_irrelevant_instance
from agentless.fl.retrieve import retrieve_locs
from agentless.repair.repair import repair, post_process_repair
from agentless.repair.rerank import normalize_patches, _load_results, majority_voting
from agentless.test.generate_reproduction_tests import post_process_tests, normalize_tests, \
    test_selection, gen_test
from agentless.test.run_regression_tests import _run_regression
from agentless.test.run_reproduction_tests import _run_reproduction_tests
from agentless.test.select_regression_tests import select_tests
from agentless.util.arguments import LocalizationArgs, RetrievalArgs, CombineArgs, RepairArgs, RegressionTestsArgs, \
    SelectRegressionTestsArgs, GenerateTestArgs, RunReproductionTestsArgs, RerankArgs
from agentless.util.utils import load_jsonl
from tqdm import tqdm

from nemo_skills.inference.generate import GenerationTask, GenerateSolutionsConfig
from nemo_skills.utils import setup_logging, get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class AgentlessConfig(GenerateSolutionsConfig):
    azure_openai_endpoint: str = ''
    openai_api_version: str = ''
    openai_api_key: str = ''
    preprocessed_data_path: str = ''


class AgentlessGenerationTask(GenerationTask):
    def __init__(self, cfg: AgentlessConfig):
        """Initializes the task, thread pool, and a dictionary to track async processes."""
        super().__init__(cfg)
        self.executor = ThreadPoolExecutor(max_workers=512)

        os.environ['AZURE_OPENAI_ENDPOINT'] = cfg.azure_openai_endpoint
        os.environ['AZURE_OPENAI_API_VERSION'] = cfg.openai_api_version
        os.environ['OPENAI_API_KEY'] = "EMPTY"
        os.environ['AZURE_OPENAI_API_KEY'] = cfg.openai_api_key
        os.environ['PROJECT_FILE_LOC'] = cfg.preprocessed_data_path

        self.async_processes = {}

    def _create_directories(self, args, output_folder, logdir=None):
        os.makedirs(output_folder, exist_ok=True)
        if logdir:
            os.makedirs(os.path.join(output_folder, logdir), exist_ok=True)

        # write the arguments
        with open(f"{output_folder}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)

    def _localize_suspicious_files(self, data_point, data, save_dir):
        args = LocalizationArgs(
            output_folder=os.path.join(save_dir, "file_level/"),
            file_level=True,
            num_threads=10,
            skip_existing=True,
        )
        args.output_file = os.path.join(args.output_folder, args.output_file)
        self._create_directories(
            args=args,
            output_folder=args.output_folder,
            logdir="localization_logs/"
        )

        localize_instance(data_point, args, data, start_file_locs=None, existing_instance_ids=set())

    def _remove_irrelevant_folders(self, data_point, data, save_dir):
        args = LocalizationArgs(
            output_folder=os.path.join(save_dir, "file_level_irrelevant/"),
            file_level=True,
            irrelevant=True,
            num_threads=10,
            skip_existing=True,
        )
        args.output_file = os.path.join(args.output_folder, args.output_file)
        self._create_directories(
            args=args,
            output_folder=args.output_folder,
            logdir="localization_logs/"
        )
        localize_irrelevant_instance(data_point, args, data, existing_instance_ids=set())

    def _retrieve_from_relevant_folders(self, data_point, data, save_dir):
        # causes an error internally if this exists already.
        persist_dir = os.path.join(save_dir, "embedding/")
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        args = RetrievalArgs(
            index_type="simple",
            filter_type="given_files",
            filter_file=os.path.join(save_dir, "file_level_irrelevant/loc_outputs.jsonl"),
            output_folder=os.path.join(save_dir, "retrieval_embedding/"),
            persist_dir=persist_dir,
            num_threads=10,
        )
        args.output_file = os.path.join(args.output_folder, args.output_file)
        self._create_directories(
            args=args,
            output_folder=args.output_folder,
            logdir="retrieval_logs/"
        )
        found_files = load_jsonl(args.filter_file) if args.filter_file else []
        retrieve_locs(data_point, args, data, found_files, prev_o=[], write_lock=None)

    def _merge_localizations(self, save_dir):
        args = CombineArgs(
            retrieval_loc_file=os.path.join(save_dir, "retrieval_embedding/retrieve_locs.jsonl"),
            model_loc_file=os.path.join(save_dir, "file_level/loc_outputs.jsonl"),
            top_n=3,
            output_folder=os.path.join(save_dir, "file_level_combined/")
        )
        args.output_file = os.path.join(args.output_folder, args.output_file)
        self._create_directories(
            args=args,
            output_folder=args.output_folder,
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
        args.output_file = os.path.join(args.output_folder, args.output_file)
        self._create_directories(
            args=args,
            output_folder=args.output_folder,
            logdir="localization_logs/"
        )
        start_file_locs = load_jsonl(args.start_file) if args.start_file else None
        localize_instance(data_point, args, data, start_file_locs=start_file_locs, existing_instance_ids=set())

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
        args.output_file = os.path.join(args.output_folder, args.output_file)
        self._create_directories(
            args=args,
            output_folder=args.output_folder,
            logdir="localization_logs/"
        )
        start_file_locs = load_jsonl(args.start_file) if args.start_file else None
        localize_instance(data_point, args, data, start_file_locs=start_file_locs, existing_instance_ids=set())

    def _separate_edit_locations(self, save_dir):
        args = LocalizationArgs(
            merge=True,
            output_folder=os.path.join(save_dir, "edit_location_individual/"),
            top_n=3,
            num_samples=4,
            start_file=os.path.join(save_dir, "edit_location_samples/loc_outputs.jsonl"),
        )
        args.output_file = os.path.join(args.output_folder, args.output_file)
        self._create_directories(
            args=args,
            output_folder=args.output_folder,
            logdir="localization_logs/"
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
            self._create_directories(
                args=args,
                output_folder=args.output_folder,
                logdir="repair_logs/"
            )
            args.output_file = os.path.join(args.output_folder, "output.jsonl")

            if args.post_process:
                args.raw_output_file = args.output_file
                if args.select_id == -1:
                    args.output_file = args.raw_output_file.replace(
                        ".jsonl", "_processed.jsonl"
                    )
                else:
                    args.output_file = args.raw_output_file.replace(
                        ".jsonl", f"_{args.select_id}_processed.jsonl"
                    )
                post_process_repair(args)
            elif args.gen_and_process:
                repair(args)
                args.raw_output_file = args.output_file
                for i in range(args.max_samples):
                    args.output_file = args.raw_output_file.replace(
                        ".jsonl", f"_{i}_processed.jsonl"
                    )
                    args.select_id = i
                    post_process_repair(args)
            else:
                repair(args)

    def _prepare_regression_tests(self, save_dir, instance_id):
        run_args = RegressionTestsArgs(
            run_id="generate_regression_tests",
            output_file=os.path.join(save_dir, "passing_tests.jsonl"),
            instance_ids=[instance_id]
        )
        _run_regression(run_args)

        args = SelectRegressionTestsArgs(
            passing_tests=run_args.output_file,
            output_folder=os.path.join(save_dir, "select_regression/"),
            instance_ids=[instance_id]
        )
        args.output_file = os.path.join(args.output_folder, "output.jsonl")
        self._create_directories(
            args=args,
            output_folder=args.output_folder,
            logdir="select_test_logs/"
        )
        select_tests(args)

    def _run_regression_on_patches(self, save_dir, num_repair_samples, instance_id):
        for i in range(1, num_repair_samples + 2):
            folder_name = f"repair_sample_{i}"

            def run_single(num):
                args = RegressionTestsArgs(
                    regression_tests=os.path.join(save_dir, "select_regression/output.jsonl"),
                    predictions_path=os.path.join(save_dir, folder_name, f"output_{num}_processed.jsonl"),
                    run_id=f"{folder_name}_regression_{num}",
                    instance_ids=[instance_id],
                    num_workers=10
                )
                _run_regression(args)

            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(run_single, range(10))

    def _generate_reproduction_tests(self, save_dir, instance_id , data):
        args = GenerateTestArgs(
            max_samples=40,
            output_folder=os.path.join(save_dir, "reproduction_test_samples/"),
            num_threads=10,
        )
        args.output_file = os.path.join(args.output_folder, "output.jsonl")
        self._create_directories(
            args=args,
            output_folder=args.output_folder,
            logdir="generating_test_logs/"
        )
        gen_test(
            instance_id=instance_id,
            args=args,
            swe_bench_data=data,
            prev_o=[],
        )
        args.raw_output_file = args.output_file
        for i in range(args.max_samples):
            args.output_file = args.raw_output_file.replace(".jsonl", f"_{i}_processed_reproduction_test.jsonl")
            args.select_id = i
            post_process_tests(args)

    def _run_and_filter_reproduction_tests(self, save_dir, instance_id):
        def run_single(num):
            args = RunReproductionTestsArgs(
                run_id=f"reproduction_test_generation_filter_sample_{num}",
                test_jsonl=os.path.join(save_dir,
                                        f"reproduction_test_samples/output_{num}_processed_reproduction_test.jsonl"),
                instance_ids=[instance_id],
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
        self._create_directories(
            args=args,
            output_folder=args.output_folder,
            logdir="generating_test_logs/"
        )
        normalize_tests(args)
        test_selection(args)

    def _evaluate_patches_with_repro_tests(self, save_dir, num_repair_samples, instance_id):
        for i in range(num_repair_samples + 2):
            run_id_prefix = f"repair_sample_{i}"

            def run_single(num):
                args = RunReproductionTestsArgs(
                    test_jsonl="results/swe-bench-lite/reproduction_test_samples/reproduction_tests.jsonl",
                    predictions_path=os.path.join(save_dir, f"{run_id_prefix}/output_{num}_processed.jsonl"),
                    run_id=f"{run_id_prefix}_reproduction_{num}",
                    instance_ids=[instance_id],
                    num_workers=10
                )
                _run_reproduction_tests(args)

            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(run_single, range(10))

    def _rerank_and_select_final_patch(self, save_dir, save_file, num_repair_samples):
        patch_folders = ','.join(
            [os.path.join(save_dir, f"repair_sample_{i}") for i in range(1, num_repair_samples + 2)])
        args = RerankArgs(
            patch_folder=patch_folders,
            num_samples=40,
            deduplicate=True,
            regression=True,
            reproduction=True,
            output_file=os.path.join(save_dir, save_file)
        )
        normalize_patches(args)
        _load_results(args)
        majority_voting(args)

    def _run_agentless(self, data_point, data):
        """Runs the full agentless pipeline for a single data point."""
        base_dir = os.path.splitext(self.cfg.output_file)[0] + "/"
        save_dir = os.path.join(base_dir, data_point['instance_id'])
        os.makedirs(save_dir, exist_ok=True)
        num_additional_repair_samples = 3

        # 1. Localize suspicious files using LLM
        self._localize_suspicious_files(data_point, data, save_dir)

        # 2. Remove irrelevant folders before running embedding-based retrieval localization.
        self._remove_irrelevant_folders(data_point, data, save_dir)

        # 3. Retrieval from relevant folders, filtering out irrelevant files.
        self._retrieve_from_relevant_folders(data_point, data, save_dir)

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
        self._prepare_regression_tests(save_dir, instance_id=data_point['instance_id'])

        # 11. Run the selected tests on the generated repair patches.
        self._run_regression_on_patches(save_dir, num_additional_repair_samples, instance_id=data_point['instance_id'])

        # 12. Generate reproduction tests to see if it solves the original issues using LLM.
        self._generate_reproduction_tests(save_dir, instance_id=data_point['instance_id'], data=data)

        # 13. Run reproduction tests to see if they can reproduce the issue, and filter those that do not.
        self._run_and_filter_reproduction_tests(save_dir, instance_id=data_point['instance_id'])

        # 14. Apply majority voting to select one reproduction test per issue.
        self._select_final_reproduction_test(save_dir)

        # 15. Evaluate generated patches using selected reproduction test.
        self._evaluate_patches_with_repro_tests(save_dir, num_additional_repair_samples, instance_id=data_point['instance_id'])

        # 16. Perform re-ranking using the regression/reproduction test results to select final patch.
        self._rerank_and_select_final_patch(
            save_dir,
            save_file='all_preds.jsonl',
            num_repair_samples=num_additional_repair_samples
        )
        # todo: remove this once we've managed to get a sample working.
        raise ValueError
        return {"completed": True, 'generation': os.path.join(save_dir, 'all_preds.jsonl')}

    def get_llm_generations(self, requests_in_progress, generations):
        """Checks for and retrieves results from completed asynchronous runs."""
        for idx, instance_id in requests_in_progress.items():
            if instance_id in self.async_processes:
                future = self.async_processes[instance_id]
                if future.done():
                    generations[idx] = future.result()
        return requests_in_progress, generations

    def llm_generate(self, data_points, data, is_async=False):
        """
        Generates results either synchronously or asynchronously.
        - Sync: Processes all data points and returns a list of results.
        - Async: Submits all data points to the thread pool and returns their instance_ids.
        """
        base_dir = os.path.splitext(self.cfg.output_file)[0] + "/"
        os.makedirs(base_dir, exist_ok=True)

        if not is_async:
            outputs = []
            for data_point in tqdm(data_points, desc="Remaining generations"):
                outputs.append(self._run_agentless(data_point, data))
            return outputs
        else:
            instance_ids = []
            for data_point in data_points:
                instance_id = data_point['instance_id']
                future = self.executor.submit(self._run_agentless, data_point, data)
                self.async_processes[instance_id] = future
                instance_ids.append(instance_id)
            return instance_ids


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_agentless_config", node=AgentlessConfig)

GENERATION_TASK_CLASS = AgentlessGenerationTask


@hydra.main(version_base=None, config_name='base_agentless_config')
def agentless(cfg: AgentlessConfig):
    cfg = AgentlessConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = AgentlessGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    setup_logging()
    agentless()
