pip install \
    torch==2.9.0+cu130 \
    torchaudio==2.9.0+cu130 \
    torchvision==0.24.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

# Install NeMo and other dependencies
pip install \
    "nemo-toolkit[asr]==2.5.3" \
    soundfile==0.13.1 \
    librosa==0.11.0

# Install transformers and dependencies for LLM
pip install  \
    transformers==4.57.1 \
    accelerate==1.12.0 \
    bitsandbytes==0.46.0 \
    sentencepiece>=0.2.1 \
    protobuf==5.29.5 \
    datasets==4.4.1 \
    peft>=0.11.0

pip install opencv-python-headless==4.10.0.84 \
    diffusers==0.30.2 \
    einops==0.8.1 \
    sounddevice==0.5.3 

pip install gdown
pip install coloredlogs flatbuffers numpy packaging protobuf sympy
pip install -U --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu
pip install -U onnx onnxscript

python3 ./download_data.py
python3 ./onnx_csm/export_backbone_step_onnx.py
python3 ./onnx_csm/export_depth_decoder_onnx.py
python3 ./onnx_csm/generate_with_backbone_and_depth_onnx.py

mv ./local_models /local_models

mv ./testing_avatar_creation /local_models/musetalk_model/testing_avatar_creation

ln -s /local_models ./local_models

python3 model_repository/tts/1/tts_generator.py