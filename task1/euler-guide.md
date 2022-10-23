How to run task 1 on Euler
==========================
This guide describes how you can run task 1 on the ETH Euler cluster.
Please only follow this approach if you are unable to run the task on your local machine,
e.g., if you own an M1 MacBook or have very old hardware.
All the tasks are designed so that they can be completed on a personal laptop,
including if you are trying to obtain a competitive leaderboard score.
Almost all students will have a strictly smoother experience working with Docker
on their own laptop rather than following this guide.

**Read the "Important information" section very carefully before you start with this guide.
Failure to do so may result in loss of cluster access and/or termination of your ETH account!**

Note that you can adapt this guide to run the tasks on other systems, such as Google Colab. However, we can not provide any additional guidance or support for other approaches.


Important information
---------------------
1. **Never** perform **any computations** on Euler directly; **always use the batch system (bsub)**! If you run your solution or other heavy computation directly, you will lose access to the cluster (possibly forever).
2. Please use this approach only as a last resort to not overload the cluster.
3. This is an unofficial approach, hence we can only provide very limited support to you.
4. The [ETH Scientific and High Performance Computing Wiki](https://scicomp.ethz.ch/wiki/Main_Page) provides very detailed documentation for everything related to the cluster, as well as a detailed [FAQ](https://scicomp.ethz.ch/wiki/FAQ). Whenever you have cluster-related questions, please search the wiki first. We will not answer questions that are already answered in the wiki.
5. Your final code that you hand-in should still run via Docker. If you only change *solution.py* and do not use any hard-coded file paths, then the Docker-approach should still work. Nevertheless, please test your code with Docker before submitting it.


Initial one-time setup
----------------------
The following steps prepare the cluster for running the tasks. You need to do those steps only once for the entire course.

1. Read *and understand* the documentation of the cluster and the rules for using it:
	1. Read [Accessing the cluster](https://scicomp.ethz.ch/wiki/Accessing_the_cluster) and follow the instructions to gain access to the Euler cluster.
	2. Read the [Getting started with clusters](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters) tutorial, in particular sections 2, 3, and 5.
	3. Revisit the [ETH Zurich Acceptable Use Policy for Information and Communications Technology ("BOT")](https://rechtssammlung.sp.ethz.ch/Dokumente/203.21en.pdf).
2. Connect to the cluster and change into your home directory (`cd ~`) if necessary.
3. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (our Python distribution):
	1. Run `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`.
	2. Run `chmod +x Miniconda3-latest-Linux-x86_64.sh`.
	3. Run `./Miniconda3-latest-Linux-x86_64.sh`.
	4. Review and accept the license agreement.
	5. Make sure the installation location is `/cluster/home/USERNAME/miniconda3` where `USERNAME` is your ETH username and press *ENTER* to confirm the location.
	6. When asked whether you wish the installer to initialize Miniconda3, answer `yes`.
	7. Disconnect and reconnect to the cluster.
	8. Run `conda config --set auto_activate_base false`.
	9. Run `rm Miniconda3-latest-Linux-x86_64.sh` to remove the installer.


Per-task setup
--------------
You need to perform the following steps only once for task 1, but again for future tasks.

1. Upload the extracted handout to the cluster and store it as *~/task1/*. Your *solution.py* file should be stored as *~/task1/solution.py*.
2. Connect to the cluster and change into the task directory by running `cd ~/task1/`.
3. Create the task's Python environment:
    1. Run `conda deactivate` to make sure that you are starting from a clean state.
    2. Run `conda create -n pai-task1 python=3.8.*` and confirm the prompts with *y*.
    3. Run `conda activate pai-task1`.
    4. Run `python --version` and make sure that your Python version starts with *3.8*.
    5. Run `pip install -U pip && pip install -r requirements.txt`
    6. Whenever you change *requirements.txt*, do the following:
        1. Always run `conda activate pai-task1` and make sure the environment is activated *before* running any `pip` commands!
        2. If you added a new package, you need to re-run `pip install -U pip && pip install -r requirements.txt`.
        3. If you removed a package, you need to run `pip uninstall PACKAGE` where `PACKAGE` is the package name.
4. After finishing task 1, you can free some space by removing the environment via `conda env remove -n pai-task1`.


Running your code
-----------------
You need to perform the following steps each time you reconnect to the cluster and want to run your solution.

**Only run your solution via the batch system, and never directly! If you run your solution directly, you will lose access to the cluster now and in the future!**

1. Preparation:
	1. Connect to the cluster and change into the task directory by running `cd ~/task1/`.
	2. Run `conda activate pai-task1`.
2. Submit a batch job:
    1. Run

            bsub -o "$(pwd)/logs.txt" -n 4 -R "rusage[mem=2048]" python -u checker_client.py --results-dir "$(pwd)"

    2. Do **not** modify any files in the *~/task1/* directory until the batch job has completed. The cluster uses your code at the time of *execution*, not at the time of *submission*!
    3. You can inspect the state of your batch job by running `bbjobs`.
    4. After your job starts, you can determine its job ID via `bbjobs` and then watch it in real-time using `bpeek -f jobID` where your replace `jobID` with the job's ID.
    5. As soon as your job completed, you can find its output in *~/task1/logs.txt* and your submission in *~/task1/results_check.byte*.
	Note that the batch system appends the outputs of multiple jobs to the log file.
