# Ultrasound Analyzer
Analyzes the ultrasound images and creates a medical report using EchoVLM-V2

## Device under Test
Processor: Intel® Core™ Ultra 7 165U   
Integrated GPU(iGPU): Intel® Graphics   
NPU: Intel® AI Boost   
Memory: 32GB  
OS: Windows 11 Enterprise 24H2   
Python: 3.12.3   

## Prerequisites
### Model Optimization    
1. Create a separate Python virtual environment to install the dependencies
```
pip install "optimum-intel[openvino]==1.26.1" "hugginface_hub[hf_xet]"   
```
2. Compress the model weights into INT4 precision
```
optimum-cli export openvino --model chaoyinshe/EchoVLM_V2_lingshu_base_7b_instruct_preview –weight-format int4 --sym --ratio 1.0 --group-size 128 echo_vlm_v2_int4
```
### Model Serving   
1. Open Windows Command Prompt and download [OpenVINO Model Server](https://docs.openvino.ai/2026/model-server/ovms_docs_deploying_server_baremetal.html)
```
curl -L https://github.com/openvinotoolkit/model_server/releases/download/v2026.0/ovms_windows_python_on.zip -o ovms.zip   
tar -xf ovms.zip
```
2. Activate the environment variables
```
cd ovms
.\setupvars.bat
```
3. Serve the model
```
ovms --port 9000 --model_name EchoVLM_V2 --model_path <echo_vlm_v2_int4> --rest_port 8000 --log_level DEBUG --rest_workers 2   
```
Note: Make sure the path to the compressed model is valid    
4. Open Windows PowerShell to test the model response   
```
(Invoke-WebRequest -Uri "http://localhost:8000/v3/chat/completions" `
 -Method POST `
 -Headers @{ "Content-Type" = "application/json" } `
 -Body '{"model": "EchoVLM_V2", "max_tokens": 30, "temperature": 0, "stream": false, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are the 3 main tourist attractions in Paris?"}]}').Content
```

### Test Dataset   
Download the test data from [Breast Ultrasound](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data)   

## Installation
Create a new Python virtual environment for installing the depedencies for ultrasound analyzer   
```
pip install -r requirements.txt
```
## Running the application   
```
streamlit run app.py
```
Upload input image
<img width="1769" height="1297" alt="image" src="https://github.com/user-attachments/assets/f2fef1c7-bbe9-45f5-a9aa-f7ec7d01f4b4" />

Input the prompt   
<img width="2067" height="572" alt="image" src="https://github.com/user-attachments/assets/4f62207a-5d28-4d1e-8194-0942adf3fc8a" />

