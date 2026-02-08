# Calibration-Under-Multiple-Correct-Answers
This is the official repo for "Evaluating and Calibrating LLM Confidence on Questions with Multiple Correct Answers".

## 📂 Directory Structure


```bash
├── data_construction/          # Data construction phase
│   ├── example/                # Example scripts and corresponding output
│   └── final_data/             # Final MACE benchmark and intermediate artifacts
├── Inference/                  # Model inference phase
│   ├── example/                # Inference result examples
│   ├── utils/                  # Core utility classes
│   ├── inf_pipeline_api.sh     # Inference pipeline for API-based models
│   ├── inf_pipeline_local.sh   # Inference pipeline for local models
│   └── run_MLLM.py             # Main entry point for MLLM execution
└── requirements.txt            # Environment dependency configuration
```

## 🛠️ Installation
Before running the scripts, please ensure you have the necessary environment set up. Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started
The workflow is divided into two main steps: Data Construction and Inference.

### Data Construction

The data_construction folder contains the pipeline to construct the MACE benchmark. This includes fetching entity popularity, retrieving facts, and generating QA pairs. 

- Quick Demo: To generate a sample dataset using the example pipeline, run
```bash
cd data_construction
bash example/pipeline.sh
```

- Output: The processed data ready for inference will be located in `data_construction/final_data/MACE`.

### Inference

The Inference folder provides scripts to evaluate models on the MACE benchmark. You can choose between running locally deployed models or API-based models.

- Option A: Run Local Model
Use this script if you are using local weights (e.g., Hugging Face models):

```Bash
cd Inference
bash inf_pipeline_local.sh
```

- Option B: Run API Model
Use this script for API-based models (e.g., OpenAI, DeepSeek API):

```Bash
cd Inference
bash inf_pipeline_api.sh
```

- Results: After inference, the result files will be saved in the `result/MACE/<model_name>` directory. An example path: `Inference/example/result/MACE/deepseek-v3`.
