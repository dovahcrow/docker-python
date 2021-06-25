ARG BASE_TAG=staging

FROM gcr.io/kaggle-images/python:${BASE_TAG}

ADD clean-layer.sh  /tmp/clean-layer.sh

# Ensure the cuda libraries are compatible with the custom Tensorflow wheels.
# TODO(b/120050292): Use templating to keep in sync or COPY installed binaries from it.
ENV CUDA_MAJOR_VERSION=11
ENV CUDA_MINOR_VERSION=0
ENV CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION
ENV CUDNN_VERSION=8.0.4
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/bin:${PATH}
# The stub is useful to us both for built-time linking and run-time linking, on CPU-only systems.
# When intended to be used with actual GPUs, make sure to (besides providing access to the host
# CUDA user libraries, either manually or through the use of nvidia-docker) exclude them. One
# convenient way to do so is to obscure its contents by a bind mount:
#   docker run .... -v /non-existing-directory:/usr/local/cuda/lib64/stubs:ro ...
# TODO(rosbo): Put this back
# ENV LD_LIBRARY_PATH_NO_STUBS="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
# ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH"

# Install OpenCL & libboost (required by LightGBM GPU version)
RUN apt-get install -y ocl-icd-libopencl1 clinfo libboost-all-dev && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    /tmp/clean-layer.sh

# When using pip in a conda environment, conda commands should be ran first and then
# the remaining pip commands: https://www.anaconda.com/using-pip-in-a-conda-environment/
# However, because this image is based on the CPU image, this isn't possible but better
# to put them at the top of this file to minize conflicts.
RUN conda remove --force -y pytorch torchvision torchaudio torchtext cpuonly && \
    conda install "pytorch=1.7" "torchvision=0.8" "torchaudio=0.7" "torchtext=0.8" "cudf=0.16" "cuml=0.16" cudatoolkit=$CUDA_VERSION cudnn=$CUDNN_VERSION && \
    /tmp/clean-layer.sh

# Install LightGBM with GPU
RUN pip uninstall -y lightgbm && \
    cd /usr/local/src && \
    git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout tags/v3.2.0 && \
    mkdir build && cd build && \
    cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \
    make -j$(nproc) && \
    cd /usr/local/src/LightGBM/python-package && \
    python setup.py install --precompile && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    /tmp/clean-layer.sh

# Install JAX (Keep JAX version in sync with CPU image)
RUN  pip install jax==0.2.12 jaxlib==0.1.64+cuda$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION -f https://storage.googleapis.com/jax-releases/jax_releases.html && \
     /tmp/clean-layer.sh

# Reinstall packages with a separate version for GPU support.
RUN pip uninstall -y tensorflow && \
    pip install tensorflow-gpu==2.4.1 && \
    pip uninstall -y mxnet && \
    pip install mxnet-cu$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION && \
    /tmp/clean-layer.sh

# Install GPU-only packages
RUN pip install pycuda && \
    pip install cupy-cuda$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION && \
    pip install pynvrtc && \
    # b/190622765 latest version is causing issue. nnabla fixed it in https://github.com/sony/nnabla/issues/892, waiting for new release before we can remove this pin.
    pip install pynvml==8.0.4 && \
    pip install nnabla-ext-cuda$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION && \
    /tmp/clean-layer.sh

# Re-add TensorBoard Jupyter extension patch
# b/139212522 re-enable TensorBoard once solution for slowdown is implemented.
# ADD patches/tensorboard/notebook.py /opt/conda/lib/python3.7/site-packages/tensorboard/notebook.py

# Remove the CUDA stubs.
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH_NO_STUBS"
