#! /bin/bash
python setup.py develop
pip install motmetrics==1.1.3
pip install neptune-client==0.14.2
pip install open3d
pip install transformers
pip install protobuf==3.20.3
pip install matplotlib_venn
cp bugfix/cumulative_optimizer_fix.py /opt/conda/lib/python3.8/site-packages/mmcv/runner/hooks/optimizer.py
cd ../lamtk
pip install -e .
cd ../point-cloud-reid
pip install typing-extensions==4.3.0
pip uninstall neptune
pip install neptune-client==0.14.2