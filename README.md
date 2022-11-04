# yolov7_trt
a quick solution for yolov7 tensorRT deployment
## preparation
- first make sure **tensorRT** is installed in your machine.
- install **torch** & **torchvision**
- install **torch2trt** and other requirements
```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python3 setup.py install
cd ..
pip3 install -r requirements.txt
```

## xxx.pt -> xxx.onnx -> xxx.engine(xxx.pt)

- first copy file export_ONNX_for_TRT.py to the path of yolov7 project ( [yolov7 main page](https://github.com/WongKinYiu/yolov7) )

```bash
python3 export_ONNX_for_TRT.py --weights yolov7.pt --img-size 640 --batch-size 1 --simplify --opset 10
```
and **yolov7_640x640.onnx** and **yolov7_640x640.yaml** will be generated, copy both of them to the path of this project

```bash
python3 onnx2trt.py --onnx yolov7_640x640.onnx --yaml yolov7_640x640.yaml --workspace 8 --fp16
```
and it will generate **yolov7_640x640.engine** and **yolov7_640x640.pt** for tensorRT c++ inference and python inference

## run
### python
modify some params in detect.py and then
```bash
python3 detect.py
```

### c++
modify your lib path in cpp/CMakeLists.txt and then
```bash
cd cpp
mkdir build && cd build
cmake ..
make
cd ..
```
modify params in run.py and then

```bash
python3 run.py
```

