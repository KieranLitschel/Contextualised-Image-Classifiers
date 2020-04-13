#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --partition=General_Usage
#SBATCH --gres=gpu:1 # GPUs requested
#SBATCH --cpus-per-task=1 # CPUs requested
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-10.0.130

export CUDNN_HOME=/opt/cuDNN-7.6.0.64_9.2

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMP=/disk/scratch/${STUDENT_ID}

mkdir -p ${TMP}/datasets
export DATASET_DIR=${TMP}/datasets

export JOB_ID="pre_train_embedding_yfcc"
export RANDOM_SEED="0"

export PRE_TRAINED_MODEL_DIR=/home/${STUDENT_ID}/HonorsProject/Embeddings/pre_train_embedding_yfcc/${RANDOM_SEED}
export OUTPUT_DIR=/home/${STUDENT_ID}/HonorsProject/Embeddings/fine_tuned_embedding_yfcc/${RANDOM_SEED}
mkdir -p ${OUTPUT_DIR}

date
echo "Copying data.."

rsync -uap /home/${STUDENT_ID}/HonorsProject/models ${DATASET_DIR}
rsync -uap /home/${STUDENT_ID}/HonorsProject/Embeddings/CCNN/ ${DATASET_DIR}
rsync -uap --progress /home/${STUDENT_ID}/HonorsProject/Embeddings/dataset ${DATASET_DIR}

date
echo "Finished copying data, starting training"

# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate hp

cd ${DATASET_DIR}/models
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/research:`pwd`/research/slim

cd ${DATASET_DIR}

# Start experiment

python -m scripts.pre_train_embedding_yfcc \
--oiv_dataset_dir ${DATASET_DIR}/dataset/oiv \
--oiv_human_dataset_dir ${DATASET_DIR}/dataset/oiv_human_verified \
--output_dir ${OUTPUT_DIR} \
--pre_trained_model_dir ${PRE_TRAINED_MODEL_DIR} \
--classes_encoder_path ${DATASET_DIR}/dataset/classes_encoder \
--random_seed 0 \
--max_train_time 7

date
echo "Finished"
