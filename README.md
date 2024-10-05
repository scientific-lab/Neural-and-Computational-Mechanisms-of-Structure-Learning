
# Generalization Based (GB) and Prediction Error Based (PE) Model Code

This webpage provides the Generalization Based (GB) and Prediction Error Based (PE) model codes for the article "Neural and Computational Mechanisms of Structure Learning for Associative Memory in the Prefrontal Cortex."

## Overview

The models consist of a hippocampus simulated by a Hopfield network and an Orbitofrontal Cortex (OFC) model simulated by a three-layer feedforward neural network. Weights are updated using either the GB or PE method.

## Running the GB Model

To run the GB model, execute the following command in Python:

```bash
python runmodelgb.py id
```

Where `id` is the current trial number. The model will run continuously for 5 trials, starting with the structured condition followed by the random condition for each trial.

## Running the PE Model

After the GB model has finished running, you can run the PE model by executing the following command in Python:

```bash
python runmodelpe.py id
```

This ensures that the same stimuli used in the GB model are used for the PE model.

## Instructions

1. **Download the Code**: Download the provided code from the webpage.
2. **Prepare Your Environment**: Ensure you have Python installed on your system.
3. **Run the GB Model**: Use the command mentioned above to run the GB model for a specified trial number.
4. **Run the PE Model**: After the GB model, run the PE model with the same trial number to ensure consistency in stimuli.

## Requirements

- Python (version 3.x)
- NumPy
- Matplotlib (for visualization, if needed)

## Contact

For any queries or issues related to the code, please contact the support team at [sofiazhao@foxmail.com](mailto:sofiazhao@foxmail.com).

---

