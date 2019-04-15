# What is this library?
This library contains the code needed to reproduce all experiments in the paper [Deep Learning On Code With A Graph Vocabulary](http://tensorlab.cms.caltech.edu/users/anima/pubs/Deep_Learning_On_Code_with_an_Unbounded_Vocabulary.pdf).

It's meant to be used along with [this data preprocessing library](https://github.com/mwcvitkovic/Deep_Learning_On_Code_With_A_Graph_Vocabulary--Code_Preprocessor).

# How do I run your code?
## Installation
### Python
Install the [Conda](https://conda.io/docs/index.html) python package manager.  Then follow the instructions [here](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
using the file `environment.yml` file in this library's root directory to satisfy the python requirements to run this library's code.

In theory our code is OS-agnostic, but we ran all our experiments on Ubuntu Linux, so that's where you're most likely to have installation success.

### Cloud Integration (optional)
We ran our experiments on Amazon EC2 instances.  Everything should run fine locally, but our code includes functionality to start jobs on [AWS Managed Instances](https://docs.aws.amazon.com/systems-manager/latest/userguide/what-is-systems-manager.html) and save data and logs to [S3](https://aws.amazon.com/s3/).  

To use these features, you'll need the [AWS CLI](https://aws.amazon.com/documentation/cli/) installed and configured, and you need to edit the details in the `experiments/temp.aws_config.py` file and rename it `experiments/aws_config.py`.  (Warning: `aws_config.py` is in the .gitignore since it might contain sensitive info.)

### Tests (optional)
We included as many unit tests as we could.  They're in the `tests` directory, whose directory structure mirrors that of the rest of the library.  (Warning: they take a while to run, and they expect a GPU.)

You can run them from the project root directory with `python -m unittest`.

## Training and Evaluating models
All code in this library expects to be run from the library's root directory with python running modules as scripts.  E.g. `python -m experiments.VarNaming_vocab_comparison.train_models`.

The general workflow is
1. Create some .gml files with [this library](https://github.com/mwcvitkovic/Deep_Learning_On_Code_With_A_Graph_Vocabulary--Code_Preprocessor).
2. Create a `Task` instance for the task you want the model to perform, as shown in the file `experiments/make_tasks_and_preprocess_for_experiment.py`.
3. Turn the task into preprocessed datapoints by running `python -m preprocess_task_for_model`.
4. Run `python -m train_model_on_task`.
5. Run `python -m evaluate_model` to see how your model did on a test set.

## Recreating the experiments in the paper
1. Use [this library](https://github.com/mwcvitkovic/Deep_Learning_On_Code_With_A_Graph_Vocabulary--Code_Preprocessor) with its existing `repositories.txt` file to download 18 maven repositories and preprocess their contents into Augmented ASTs.
2. Move the directories produced via step 1. to `s3shared/18_popular_mavens/repositories`.  (Don't worry if you're not using S3 - it'll still work.)
3. Navigate to `s3shared/18_popular_mavens/` and run `experiments/make_train_test_split.sh` from the command line.
4. For either the Fill In The Blank experiment (`FITB_vocab_comparison`) or the Variable Naming experiment (`VarNaming_vocab_comparison`), run `python -m experiments.<experiment name>.make_tasks_and_preprocess`.  (You may need to change some args/kwargs in this file to suit your setup, e.g. changing `aws_config['remote_ids']['box1']` to `'local'` if you want to run locally.)
5. Run `python -m experiments.<experiment name>.train_models`. (Again, you may need to change some args/kwargs in this file to suit your setup.)
6. Run `python -m experiments.<experiment name>.evaluate_models`. (Again, you may need to change some args/kwargs in this file to suit your setup.)

# Questions?
Feel free to get in touch with [Milan Cvitkovic](mailto:mwcvitkovic@gmail.com) or any of the other paper authors.  We'd love to hear from you!
