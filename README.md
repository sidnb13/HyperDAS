# HyperDAS

## Setup

1. Create and activate the Conda environment:
   ```
   conda create -n hyperdas python=3.8
   conda activate hyperdas
   ```

2. Install the required packages:
   
   Option 1: Using requirements.txt
   ```
   pip install -r requirements.txt
   ```
   
   Option 2: Using environment.yml (if available)
   ```
   conda env update -f environment.yml
   ```

3. Set up environment variables:
   - Copy the `.env.example` file to `.env`
   - Edit the `.env` file and set your HuggingFace token:
     ```
     HF_TOKEN="your_huggingface_token_here"
     ```

4. Install pre-commit hooks:
   ```
   pre-commit install
   # Optional, to run on all files right now
   pre-commit run --all-files
   ```

## Usage

1. Activate the Conda environment:
   ```
   conda activate hyperdas
   ```

2. Run the main training script:
   ```
   python train.py
   ```

   Or use the baseline training script:
   ```
   python train_baseline.py
   ```

3. To use Hydra for configuration management, ensure `USE_HYDRA=true` in your `.env` file. You can then override config values:
   ```
   python train.py model_name_or_path=/path/to/your/model batch_size=32
   ```

4. For more detailed configuration options, refer to the `config/config.yaml` file.

## Docker Usage

To use HyperDAS with Docker:

1. Ensure you have Docker and Docker Compose installed on your system.

2. Build the Docker image:
   ```
   docker-compose build
   ```

3. Start the Docker container:
   ```
   docker-compose up -d
   ```

4. To access the container's shell:
   ```
   docker exec -it hyperdas /bin/bash
   ```

5. Once inside the container, you can run the training scripts as described in the Usage section above.

6. To stop the container:
   ```
   docker-compose down
   ```

Note: The Docker setup includes GPU support and mounts the necessary volumes for seamless development and data persistence.