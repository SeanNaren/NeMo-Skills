import os
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

from git import Repo
from tqdm import tqdm

from .logger import logger


class RepositoryManager:
    """Handles repository operations with multiprocessing support."""

    # Class variable to store data_dir for static methods
    _data_dir = None

    def __init__(self, config=None, dataset_hash=None):
        """Initialize the repository manager.

        Args:
            config: Optional Config object containing configuration parameters.
                   If not provided, uses default data_dir.
            dataset_hash: Hash of the dataset configuration for directory organization.
        """
        if config is None:
            raise ValueError("Config object is required")

        # Use dedicated repos directory under config.data_dir with dataset hash
        self.data_dir = Path(config.data_dir) / "repos" / f"dataset_{dataset_hash}"
        # Set class variable for static methods to base data directory (without dataset name)
        RepositoryManager._data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Store config for use in tree generation
        self.config = config

    def download_repository(self, instance_id: str, repo_name: str, base_commit: str) -> str:
        """Download and checkout repository at specific commit."""
        logger().verbose(f"        Setting up repository: {repo_name}")

        repo_dir = self.data_dir / instance_id
        repo_url = f"https://github.com/{repo_name}.git"

        # Check if repository exists and has content
        repo_exists = repo_dir.exists() and any(repo_dir.iterdir())

        if not repo_exists:
            logger().info(f"        Cloning repository from {repo_url}")

            try:
                Repo.clone_from(repo_url, repo_dir)
                logger().success(f"Repository cloned successfully")
            except Exception as e:
                logger().error(f"Failed to clone repository: {e}")
                raise
        else:
            logger().verbose(f"        Repository already exists at {repo_dir}")

        # Initialize git repo object
        try:
            repo = Repo(repo_dir)
        except Exception as e:
            logger().error(f"Failed to initialize git repository at {repo_dir}: {e}")
            logger().info("        Removing corrupted repository and retrying...")
            # If repo is corrupted, remove it and try cloning again
            shutil.rmtree(repo_dir)
            Repo.clone_from(repo_url, repo_dir)
            repo = Repo(repo_dir)

            logger().verbose(f"        Checking if commit {base_commit} exists...")
        try:
            # Try to find the commit
            commit_obj = repo.commit(base_commit)
            logger().verbose(f"        Commit found: {commit_obj.hexsha[:8]} - {commit_obj.message.split()[0]}")
        except Exception as e:
            logger().warning(f"Commit {base_commit} not found in repository")
            logger().verbose("        Available recent commits:")
            try:
                # Show recent commits to help debugging
                for commit in repo.iter_commits("HEAD", max_count=5):
                    logger().verbose(f"          {commit.hexsha[:8]} - {commit.message.split()[0]}")
            except Exception:
                logger().warning("Could not retrieve recent commits")

            # Try to fetch more commits from remote
            logger().info("        Fetching from remote to get more commits...")
            try:
                repo.git.fetch("--all")
                # Try again after fetch
                commit_obj = repo.commit(base_commit)
                logger().verbose(f"        Commit found after fetch: {commit_obj.hexsha[:8]} - {commit_obj.message.split()[0]}")
            except Exception as fetch_error:
                logger().error(f"Commit {base_commit} still not found after fetch: {fetch_error}")
                raise ValueError(f"Commit {base_commit} does not exist in repository {repo_name}")

        # Now checkout the commit
        logger().verbose(f"        Checking out commit: {base_commit}")
        try:
            repo.git.checkout(base_commit)
            logger().success(f"Repository ready at {repo_dir}")
        except Exception as e:
            logger().error(f"Failed to checkout commit {base_commit}: {e}")
            # Try to get current HEAD for debugging
            try:
                current_head = repo.head.commit.hexsha[:8]
                logger().verbose(f"        Current HEAD is at: {current_head}")
            except Exception:
                logger().warning("Could not determine current HEAD")
            raise

        return str(repo_dir)

    def get_repository_tree(self, repo_dir: str) -> str:
        """Generate repository tree structure with line counts."""
        return self._generate_tree_with_line_counts(repo_dir)

    def _generate_tree_with_line_counts(self, repo_dir: str) -> str:
        """Generate tree structure with line counts by walking the directory."""
        result = []
        repo_name = os.path.basename(repo_dir)
        result.append(f"{repo_name}/")

        # Get all allowed files with their line counts first (only if show_line_counts is enabled)
        file_line_counts = {}
        if getattr(self.config, "show_line_counts", True):
            for root, dirs, files in os.walk(repo_dir):
                for file in files:
                    # Check if file has any of the allowed extensions
                    file_ext = os.path.splitext(file)[1].lstrip(".")
                    if file_ext in self.config.DEFAULT_CODE_EXTENSIONS or file in self.config.DEFAULT_CODE_EXTENSIONS:
                        full_path = os.path.join(root, file)
                        try:
                            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                                line_count = sum(1 for _ in f)
                            file_line_counts[full_path] = line_count
                        except Exception:
                            file_line_counts[full_path] = 0

        def add_to_tree(path, prefix="", rel_path=""):
            try:
                # Get all items and sort them (directories first, then files)
                items = []
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        items.append((item, True))  # (name, is_dir)
                    else:
                        # Check if file has any of the allowed extensions
                        file_ext = os.path.splitext(item)[1].lstrip(".")
                        if file_ext in self.config.DEFAULT_CODE_EXTENSIONS or item in self.config.DEFAULT_CODE_EXTENSIONS:
                            items.append((item, False))  # (name, is_file)

                # Sort: directories first, then files, both alphabetically
                items.sort(key=lambda x: (not x[1], x[0].lower()))

                for i, (item, is_dir) in enumerate(items):
                    # Skip unwanted directories and files
                    if item.startswith(".") or item in ["__pycache__", ".git"]:
                        continue

                    # Skip excluded directories
                    exclude_dirs = getattr(self.config, "exclude_dirs", set())
                    if is_dir and item.lower() in exclude_dirs:
                        continue

                    item_path = os.path.join(path, item)
                    is_last = i == len(items) - 1

                    if is_dir:
                        # Directory
                        result.append(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
                        new_prefix = prefix + ("    " if is_last else "│   ")
                        new_rel_path = os.path.join(rel_path, item) if rel_path else item
                        add_to_tree(item_path, new_prefix, new_rel_path)
                    else:
                        # Allowed file
                        full_path = os.path.join(repo_dir, rel_path, item) if rel_path else os.path.join(repo_dir, item)
                        if getattr(self.config, "show_line_counts", True):
                            line_count = file_line_counts.get(full_path, 0)
                            result.append(f"{prefix}{'└── ' if is_last else '├── '}{item} ({line_count} lines)")
                        else:
                            result.append(f"{prefix}{'└── ' if is_last else '├── '}{item}")

            except Exception:
                pass

        add_to_tree(repo_dir)
        return "\n".join(result)


    @staticmethod
    def clone_single_repo_static(args_tuple) -> Dict:
        """Static method for multiprocessing - unpacks the tuple."""
        instance_id, repo_name, base_commit, datasets_hash = args_tuple

        # Use class variable for data_dir
        if RepositoryManager._data_dir is None:
            raise ValueError("RepositoryManager._data_dir not initialized")

        try:
            # Create proper directory structure using dataset hash
            dataset_dir = RepositoryManager._data_dir / "repos" / f"dataset_{datasets_hash}"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            repo_dir = dataset_dir / instance_id
            repo_url = f"https://github.com/{repo_name}.git"

            # Check if repository exists and has content
            repo_exists = repo_dir.exists() and any(repo_dir.iterdir())

            if not repo_exists:
                Repo.clone_from(repo_url, repo_dir)
            else:
                # If repo exists, just fetch to ensure we have all commits
                repo = Repo(repo_dir)
                repo.git.fetch("--all")

            # Initialize git repo object
            repo = Repo(repo_dir)

            # Checkout the specific commit
            try:
                repo.git.checkout(base_commit)
            except Exception as e:
                # Try to fetch and checkout again
                repo.git.fetch("--all")
                repo.git.checkout(base_commit)

            return {
                "instance_id": instance_id,
                "status": "success",
                "repo_dir": str(repo_dir),
            }
        except Exception as e:
            return {"instance_id": instance_id, "status": "failed", "error": str(e)}

    def get_unique_repositories(self, dataset) -> Dict[str, List[tuple]]:
        """Get unique repositories and their associated instances from the dataset."""
        repo_instances = {}

        for task_instance in dataset["merged"]:
            repo_name = task_instance["repo"]
            instance_id = task_instance["instance_id"]
            base_commit = task_instance["base_commit"]

            if repo_name not in repo_instances:
                repo_instances[repo_name] = []
            repo_instances[repo_name].append((instance_id, base_commit))

        return repo_instances

    def clone_unique_repositories(
        self,
        unique_repos: Dict[str, List[tuple]],
        dataset_hash: str = None,
    ) -> Dict[str, str]:
        """Clone only unique repositories (one per repo name)."""
        logger().info("📥 Cloning unique repositories")
        logger().info(f"        Found {len(unique_repos)} unique repositories")

        repo_results = []
        clone_tasks = []

        # Use the provided dataset hash
        for repo_name, instances in unique_repos.items():
            # Just clone the repo - we'll checkout specific commits later in each copy
            clone_tasks.append((repo_name, None, dataset_hash))  # No specific commit needed for base repo

        if clone_tasks:
            # Use a reasonable number of processes (4-6) instead of all CPU cores
            num_processes = min(6, len(clone_tasks))
            logger().info(f"        Cloning {len(clone_tasks)} unique repositories using {num_processes} processes")
            with Pool(processes=num_processes) as pool:
                # Use imap for better progress tracking
                for result in tqdm(
                    pool.imap(self._clone_unique_repo_static, clone_tasks),
                    total=len(clone_tasks),
                    desc="📥 Cloning unique repositories",
                ):
                    repo_results.append(result)
                    if result["status"] == "success":
                        logger().verbose(f"        ✓ Repository {result['repo_name']} ready")
                    else:
                        logger().error(f"❌ Repository {result['repo_name']} failed: {result['error']}")

        # Check for failed repositories
        failed_repos = [r for r in repo_results if r["status"] == "failed"]
        if failed_repos:
            logger().warning(f"{len(failed_repos)} repositories failed to clone:")
            for failed in failed_repos:
                logger().warning(f"        - {failed['repo_name']}: {failed['error']}")

        successful_repos = {r["repo_name"]: r["repo_dir"] for r in repo_results if r["status"] == "success"}

        # Ensure all base repositories have all commits needed
        logger().info("        Ensuring base repositories have all required commits...")
        for repo_name, repo_dir in successful_repos.items():
            try:
                repo = Repo(repo_dir)
                # Fetch all commits to ensure we have everything
                repo.git.fetch("--all")
                logger().verbose(f"        ✓ Fetched all commits for {repo_name}")
            except Exception as e:
                logger().warning(f"Could not fetch all commits for {repo_name}: {e}")

        logger().success(f"Successfully cloned {len(successful_repos)} unique repositories")

        return successful_repos

    @staticmethod
    def _clone_unique_repo_static(args_tuple) -> Dict:
        """Static method for multiprocessing - clones a unique repository."""
        repo_name, base_commit, datasets_hash = args_tuple

        # Use class variable for data_dir
        if RepositoryManager._data_dir is None:
            raise ValueError("RepositoryManager._data_dir not initialized")

        try:
            # Create a temporary instance ID for the base repository
            temp_instance_id = f"temp_{repo_name.replace('/', '_')}"

            if base_commit is None:
                # Just clone the repo without checking out to any specific commit
                # Put temp dirs under the dataset directory
                dataset_dir = RepositoryManager._data_dir / "repos" / f"dataset_{datasets_hash}"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                repo_dir = dataset_dir / temp_instance_id
                repo_url = f"https://github.com/{repo_name}.git"

                if not repo_dir.exists() or not any(repo_dir.iterdir()):
                    Repo.clone_from(repo_url, repo_dir)
                else:
                    # If repo exists, just fetch to ensure we have all commits
                    repo = Repo(repo_dir)
                    repo.git.fetch("--all")

                return {
                    "repo_name": repo_name,
                    "status": "success",
                    "repo_dir": str(repo_dir),
                }
            else:
                # Clone directly without creating RepositoryManager instance
                # Put temp dirs under the dataset directory
                dataset_dir = RepositoryManager._data_dir / "repos" / f"dataset_{datasets_hash}"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                repo_dir = dataset_dir / temp_instance_id
                repo_url = f"https://github.com/{repo_name}.git"

                if not repo_dir.exists() or not any(repo_dir.iterdir()):
                    Repo.clone_from(repo_url, repo_dir)
                else:
                    # If repo exists, just fetch to ensure we have all commits
                    repo = Repo(repo_dir)
                    repo.git.fetch("--all")

                # Initialize git repo object and checkout the specific commit
                repo = Repo(repo_dir)
                try:
                    repo.git.checkout(base_commit)
                except Exception as e:
                    # Try to fetch and checkout again
                    repo.git.fetch("--all")
                    repo.git.checkout(base_commit)

                return {
                    "repo_name": repo_name,
                    "status": "success",
                    "repo_dir": str(repo_dir),
                }
        except Exception as e:
            return {"repo_name": repo_name, "status": "failed", "error": str(e)}

    def verify_repository_commit(self, repo_dir: str, expected_commit: str, instance_id: str = None) -> bool:
        """Verify that a repository is checked out at the expected commit."""
        try:
            repo = Repo(repo_dir)
            current_commit = repo.head.commit.hexsha

            # Check if the current commit matches the expected commit
            if current_commit.startswith(expected_commit) or expected_commit.startswith(current_commit):
                logger().verbose(f"        ✓ {instance_id or 'Repository'} verified at commit {current_commit[:8]}")
                return True
            else:
                logger().error(f"{instance_id or 'Repository'} at wrong commit: expected {expected_commit[:8]}, got {current_commit[:8]}")
                return False
        except Exception as e:
            logger().error(f"Failed to verify {instance_id or 'repository'}: {e}")
            return False

    def verify_all_repositories(self, successful_repos: Dict[str, str], dataset=None) -> Dict[str, str]:
        """Verify all repositories are at the correct commits."""
        logger().info("🔍 Verifying all repositories are at correct commits")
        verified_repos = {}

        # Create a mapping from instance_id to expected commit
        instance_commits = {}
        for task_instance in dataset["merged"]:
            instance_commits[task_instance["instance_id"]] = task_instance["base_commit"]

        total_repos = len(successful_repos)
        verified_count = 0

        for instance_id, repo_dir in successful_repos.items():
            if instance_id in instance_commits:
                expected_commit = instance_commits[instance_id]
                if self.verify_repository_commit(repo_dir, expected_commit, instance_id):
                    verified_repos[instance_id] = repo_dir
                    verified_count += 1
                else:
                    logger().warning(f"Repository {instance_id} failed verification - will be skipped")
            else:
                logger().warning(f"No expected commit found for {instance_id}")

        logger().success(f"Verified {verified_count}/{total_repos} repositories")
        return verified_repos

    @staticmethod
    def _copy_and_checkout_single_static(args_tuple) -> Dict:
        """Static method for multiprocessing - copy and checkout a single repository."""
        instance_id, commit_hash, base_repo_dir, data_dir = args_tuple

        # Use class variable for data_dir if not provided
        if data_dir is None:
            if RepositoryManager._data_dir is None:
                raise ValueError("RepositoryManager._data_dir not initialized")
            data_dir = str(RepositoryManager._data_dir)

        try:
            # Create instance-specific directory
            instance_repo_dir = Path(data_dir) / instance_id

            if instance_repo_dir.exists():
                # Check if the existing repo is at the correct commit
                try:
                    repo = Repo(instance_repo_dir)
                    current_commit = repo.head.commit.hexsha
                    if current_commit.startswith(commit_hash) or commit_hash.startswith(current_commit):
                        return {
                            "instance_id": instance_id,
                            "status": "success",
                            "repo_dir": str(instance_repo_dir),
                            "message": f"Already at correct commit {commit_hash[:8]}",
                        }
                    else:
                        # Remove and re-copy
                        shutil.rmtree(instance_repo_dir)
                except Exception:
                    # If there's any issue with the existing repo, remove it and re-copy
                    if instance_repo_dir.exists():
                        shutil.rmtree(instance_repo_dir)

            # Copy the base repository
            shutil.copytree(base_repo_dir, instance_repo_dir)

            # Checkout the specific commit
            repo = Repo(instance_repo_dir)
            try:
                repo.git.checkout(commit_hash)
                return {
                    "instance_id": instance_id,
                    "status": "success",
                    "repo_dir": str(instance_repo_dir),
                    "message": f"Checked out {commit_hash[:8]}",
                }
            except Exception as checkout_error:
                # Try to fetch and checkout again
                try:
                    repo.git.fetch("--all")
                    repo.git.checkout(commit_hash)
                    return {
                        "instance_id": instance_id,
                        "status": "success",
                        "repo_dir": str(instance_repo_dir),
                        "message": f"Checked out {commit_hash[:8]} after fetch",
                    }
                except Exception as fetch_error:
                    # Remove the failed copy
                    if instance_repo_dir.exists():
                        shutil.rmtree(instance_repo_dir)
                    return {
                        "instance_id": instance_id,
                        "status": "failed",
                        "error": f"Failed to checkout {commit_hash[:8]}: {fetch_error}",
                    }

        except Exception as e:
            return {"instance_id": instance_id, "status": "failed", "error": str(e)}

    def copy_and_checkout_repositories_parallel(
        self,
        unique_repos: Dict[str, List[tuple]],
        base_repo_dirs: Dict[str, str],
    ) -> Dict[str, str]:
        """Copy base repositories and checkout specific commits for each instance using multiprocessing."""
        logger().info("📋 Copying repositories and checking out specific commits (parallel)")
        instance_repo_dirs = {}

        # Prepare tasks for multiprocessing
        copy_tasks = []
        for repo_name, instances in unique_repos.items():
            if repo_name not in base_repo_dirs:
                logger().warning(f"Base repository {repo_name} not found, skipping instances")
                continue

            base_repo_dir = base_repo_dirs[repo_name]

            for instance_id, commit_hash in instances:
                copy_tasks.append((instance_id, commit_hash, base_repo_dir, str(self.data_dir)))

        total_instances = len(copy_tasks)
        logger().info(f"        Processing {total_instances} instances using {min(6, total_instances)} processes")

        if copy_tasks:
            with Pool(processes=min(6, len(copy_tasks))) as pool:
                # Use imap for better progress tracking
                for result in tqdm(
                    pool.imap(RepositoryManager._copy_and_checkout_single_static, copy_tasks),
                    total=len(copy_tasks),
                    desc="📋 Copying and checking out repositories",
                ):
                    if result["status"] == "success":
                        instance_repo_dirs[result["instance_id"]] = result["repo_dir"]
                        logger().verbose(f"        ✓ {result['instance_id']}: {result['message']}")
                    else:
                        logger().error(f"{result['instance_id']}: {result['error']}")

        logger().success(f"Successfully processed {len(instance_repo_dirs)}/{total_instances} instances")
        return instance_repo_dirs

    def copy_and_checkout_repositories(
        self,
        unique_repos: Dict[str, List[tuple]],
        base_repo_dirs: Dict[str, str],
    ) -> Dict[str, str]:
        """Copy base repositories and checkout specific commits for each instance."""
        logger().info("📋 Copying repositories and checking out specific commits")
        instance_repo_dirs = {}

        total_instances = sum(len(instances) for instances in unique_repos.values())
        processed = 0

        for repo_name, instances in unique_repos.items():
            if repo_name not in base_repo_dirs:
                logger().warning(f"Base repository {repo_name} not found, skipping instances")
                continue

            base_repo_dir = base_repo_dirs[repo_name]

            for instance_id, commit_hash in instances:
                try:
                    # Create instance-specific directory
                    instance_repo_dir = self.data_dir / instance_id

                    if instance_repo_dir.exists():
                        logger().verbose(f"        Repository {instance_id} already exists, checking if commit is correct")
                        # Check if the existing repo is at the correct commit
                        try:
                            repo = Repo(instance_repo_dir)
                            current_commit = repo.head.commit.hexsha
                            if current_commit.startswith(commit_hash) or commit_hash.startswith(current_commit):
                                logger().verbose(f"        ✓ Repository {instance_id} already at correct commit {commit_hash[:8]}")
                            else:
                                logger().verbose(f"        Repository {instance_id} at wrong commit, re-copying")
                                shutil.rmtree(instance_repo_dir)
                                raise Exception("Need to re-copy")
                        except Exception:
                            # If there's any issue with the existing repo, remove it and re-copy
                            if instance_repo_dir.exists():
                                shutil.rmtree(instance_repo_dir)
                            raise Exception("Need to re-copy")
                    else:
                        # Copy the base repository
                        logger().verbose(f"        Copying {repo_name} to {instance_id}")
                        shutil.copytree(base_repo_dir, instance_repo_dir)

                        # Checkout the specific commit
                        repo = Repo(instance_repo_dir)
                        try:
                            repo.git.checkout(commit_hash)
                            logger().verbose(f"        ✓ Checked out {commit_hash[:8]} for {instance_id}")
                        except Exception as checkout_error:
                            logger().error(f"Failed to checkout {commit_hash[:8]} for {instance_id}: {checkout_error}")
                            # Try to fetch and checkout again
                            try:
                                repo.git.fetch("--all")
                                repo.git.checkout(commit_hash)
                                logger().verbose(f"        ✓ Checked out {commit_hash[:8]} for {instance_id} after fetch")
                            except Exception as fetch_error:
                                logger().error(f"Failed to checkout even after fetch: {fetch_error}")
                                # Remove the failed copy
                                shutil.rmtree(instance_repo_dir)
                                raise fetch_error

                    instance_repo_dirs[instance_id] = str(instance_repo_dir)
                    processed += 1

                except Exception as e:
                    logger().error(f"Failed to copy/checkout {instance_id}: {e}")
                    # Continue with other instances
                    continue

        logger().success(f"Successfully processed {processed}/{total_instances} instances")
        return instance_repo_dirs

    def clone_repository_sequential(
        self,
        instance_id: str,
        repo_name: str,
        base_commit: str,
    ) -> str:
        """Clone a single repository with logging support for sequential processing."""
        return self.download_repository(instance_id, repo_name, base_commit)

    def clone_all_repositories(
        self,
        dataset,
        completed_tasks: set,
        dataset_hash: str = None,
    ) -> Dict[str, str]:
        """Clone all repositories using the optimized approach: clone unique repos + copy + checkout."""
        logger().info("📥 Cloning all repositories using optimized approach")

        # Filter out completed tasks
        tasks_to_process = []
        for task_instance in dataset["merged"]:
            instance_id = task_instance["instance_id"]
            if instance_id not in completed_tasks:
                tasks_to_process.append(task_instance)

        logger().info(f"        Total tasks in dataset: {len(dataset['merged'])}")
        logger().info(f"        Completed tasks: {len(completed_tasks)}")
        logger().info(f"        Tasks to process: {len(tasks_to_process)}")

        if not tasks_to_process:
            logger().info("        All repositories already cloned")
            return {}

        # Step 1: Get unique repositories
        unique_repos = self.get_unique_repositories({"merged": tasks_to_process})
        logger().info(f"        Found {len(unique_repos)} unique repositories")

        # Step 2: Clone unique repositories (base repos)
        base_repo_dirs = self.clone_unique_repositories(unique_repos, dataset_hash)
        if not base_repo_dirs:
            logger().error("        No base repositories could be cloned")
            return {}

        # Step 3: Copy base repositories and checkout specific commits
        if self.config.parallel_copy_checkout:
            instance_repo_dirs = self.copy_and_checkout_repositories_parallel(unique_repos, base_repo_dirs)
        else:
            instance_repo_dirs = self.copy_and_checkout_repositories(unique_repos, base_repo_dirs)

        logger().success(f"Successfully prepared {len(instance_repo_dirs)} repositories using optimized approach")
        return instance_repo_dirs
