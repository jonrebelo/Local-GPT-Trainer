## Requirements

- Python 3.7 or higher
- PyTorch
- Transformers
- Pytorch-lamb
- CUDA (optional, for GPU acceleration)
- Install Visual Studio Build Tools: Select Desktop development with C++ and .Net desktop build tools and install

## Environment:
conda install matplotlib numpy ipykernel jupyter tqdm transformers multiprocessing
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pylzma pytorch-lamb

## Instructions

### Step 1: Prepare Your Dataset
You will need to provide your own dataset. Place your dataset in a file named `data.txt` within a `training_data` directory. The dataset should be in a text format.

### Step 2: Clean Dataset
Use the provided data cleanser to clean your data and split it into training and validation sets. The data cleanser script is located in the `data_cleanser.py` file.

1. Update the file paths, pool number, and chunk size in the data cleanser script based on your dataset and hardware specifications.
2. Update the clean_text function with cleaning parameters catered to your dataset. The included function only has basic cleaning functionality and is not sufficient for all datasets. Customizing to your own data set is highly recommended.
3. Run the data cleanser script:

#### Data Split Script
The `train_val_seperator.py` script will split the data into two files: `train_split.txt` and `val_split.txt`.

### Step 3: Configure Training Parameters
Edit the main script (`main.py`) to set your training parameters such as block size, batch size, number of layers, and learning rates. Ensure your parameters match your hardware capabilities.

This did well for an RTX 3080 with 10GB of VRAM, but your mileage will vary drastically based on VRAM and GPU performance.

```bash
block_size = 128
batch_size = 24
max_iters = 25100
eval_interval = 500
eval_iters = 500
n_embd = 640
n_layer = 14
n_head = 14
dropout = 0.25
```

### Step 4: Train the Model
Run the main training script. The script will automatically handle training, evaluation, and checkpointing.

```bash
python main.py
```

#### Description

This pre-trains a GPT model with your selected dataset. Clean data is essential for optimal performance, and various optimizers and learning rates can be added to enhance the model's effectiveness.

    CUDA Training: CUDA is the preferred method for training due to its efficiency with GPU. Training on CPU is not recommended.
    Tokenizer:
        GPT_Trainer-subword: Subword tokenizer from HuggingFace.
        GPT_Trainer_c-level: Character-level encoding tokenizer.
    Parameter Customization: Parameters need to be tailored to the user's specific GPU for optimal performance.

##### Model Description

The model is a GPT-based language model utilizing multi-head attention and feed-forward neural networks for text generation. It can be fine-tuned with various learning rates and optimizers to achieve the best results.

###### Features

    Optional Layer Freezing: Freeze certain layers during fine-tuning to speed up training and potentially improve performance.
    Early Stopping: Stop training early if the model's performance ceases to improve.
    Checkpoints: Save model checkpoints during training to prevent loss of progress.
    GPT_Trainers both take advantage of tensor cores in nVidia GPUs with Pythorch's Automatic Mixed Precision (AMP) to accelerate deep learning training. Requires an nVidia RTX card for this additional accleration. 
    Learning Rate and Optimizer Iteration: Iterate through different learning rates and optimizers using a scheduler to find the best configuration.
    Data Cleanser: Data_Cleanser.py script performs basic cleaning of datasets, removing unwanted characters and formatting text.
    Training and Validation Split: train_val_seperator.py script splits datasets into training and validation sets. Ensure data is cleaned before splitting.