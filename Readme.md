# Use this commands if env variables do not write properly 
# Add them directly in our terminal 


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/jurgen/openvino/bin/intel64/Release/lib
export PYTHONPATH=$PYTHONPATH:/Users/jurgen/openvino/bin/intel64/Release/lib/python_api/python3.7

source /opt/intel/openvino/bin/setupvars.sh
# RUN like that 
https://stackoverflow.com/questions/56483931/how-to-run-environment-initialization-shell-script-from-dockerfile