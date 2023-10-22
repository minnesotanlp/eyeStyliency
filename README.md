# eyeStyliency
This repository contains the code and data for the paper [A Comparative Study on Textual Saliency of Styles from Eye Tracking, Annotations, and Language Models](https://arxiv.org/pdf/2212.09873.pdf).

# Data
In summary, the following datafiles are available:

* `data/IA_data.txt` (interest-area level data from eye tracking experiment; produced with help from the [DataViewer](https://www.sr-research.com/data-viewer/) software)
* `data/eyelink_data_normalized.csv` interest-area levle data with participant-level normalizations (z-scores) for each eye tracking metric.
* `data/scores_Accumulator.SUBTRACTIVE.csv` saliency scores based on human annotations, eye data, and integrated gradients

####Stimuli from Eye-Tracking Experiment
For the stimuli used in the experiment itself and the script to generate the individual blocks presented to participants, see the `stimuli` directory.

`calculate_ppl.py` is a one-time script used to add perplexity values to the data.

####Annotations from HummingBird
Raw human annotations are in [this repo](https://github.com/sweetpeach/hummingbird).

# Scripts and visualizations
Our processing scripts and code used to generate visualizations and other results are in `visualization_exps` directory. To run these scripts yourself, install the requirements: `pip install -r requirements.txt`.

For the openai API calls and responses, see `visualization_exps/openai_exp`.


Finally, `ia_processing_helpers` contains most procedural functions used for computing the scores.