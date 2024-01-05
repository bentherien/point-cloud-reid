#! /bin/bash
python setup.py develop
pip install motmetrics==1.1.3
pip install neptune-client
pip install open3d
pip install transformers
pip install protobuf==3.20.3
pip install matplotlib_venn
cp bugfix/cumulative_optimizer_fix.py /opt/conda/lib/python3.8/site-packages/mmcv/runner/hooks/optimizer.py
cd ../lamtk
pip install -e .
pip install typing-extensions==4.3.0