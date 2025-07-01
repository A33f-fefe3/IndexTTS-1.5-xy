用B站开源的tts合成有声书

conda create -n index-tts python=3.10
conda activate index-tts
apt-get install ffmpeg
# or use conda to install ffmpeg
conda install -c conda-forge ffmpeg

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# after conda activate index-tts
conda install -c conda-forge pynini==2.1.6
pip install WeTextProcessing --no-deps

cd index-tts
pip install -e .