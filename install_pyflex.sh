apptainer exec \
    --nv \
    --bind ${PWD}/PyFleX:/workspace/PyFleX \
    --bind ${CONDA_PREFIX}:/workspace/anaconda \
    softgym_latest.sif \
    bash -c "export PATH=/workspace/anaconda/bin:\$PATH; \
             cd /workspace/PyFleX; \
             export PYFLEXROOT=/workspace/PyFleX; \
             export PYTHONPATH=/workspace/PyFleX/bindings/build:\$PYTHONPATH; \
             export LD_LIBRARY_PATH=\$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:\$LD_LIBRARY_PATH; \
             cd bindings; mkdir -p build; cd build; \
             /usr/bin/cmake ..; make -j"


echo '# PyFleX' >> ~/.bashrc
echo "export PYFLEXROOT=${PWD}/PyFleX" >> ~/.bashrc
echo 'export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo '' >> ~/.bashrc

source ~/.bashrc