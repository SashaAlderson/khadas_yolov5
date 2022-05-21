# YOLOv5 на Khadas
## Установка Ubuntu на Khadas VIM3
1. Следуя [инструкциям](https://docs.khadas.com/linux/vim3/index.html) из раздела Quick start устанавливаем Ubuntu. Образ можно взять с [официального сайта](https://docs.khadas.com/linux/firmware/Vim3UbuntuFirmware.html) или брать уже 
протестированную версию [V1.0.7-210625](https://drive.google.com/drive/folders/1FUXloO80ecwliYHQxfVgwSeL5ceh2Hhl?usp=sharing).
## YOLOv5
1. Клонируем форк репозитория, с leakyReLU свёртками и экспортом в ONNX обрезанного слоя Detect
```
git clone https://github.com/SashaAlderson/yolov5
```
2. Обучаем сеть в течение 300 эпох следуя гайдам, либо используем [предобученную](https://drive.google.com/drive/folders/1wlErIkcGLRwXylHBNNuMXS2gnXjkixCY?usp=sharing).
## Конвертация на NPU
1. Клонируем репозиторий с готовыми бинарными файлами для конвертации, конвертировать можно на стороннем устройстве.
```
git clone https://github.com/SashaAlderson/tengine_khadas_sdk
cd tengine_khadas_sdk/tengine_tools
```
2. Устанавливаем opencv
```
sudo apt install libopencv-dev
```
3. Конвертируем модель в tmfile

  Параметры: 
```bash
$ ./convert_tool/convert_tool -h

---- Tengine Convert Tool ---- 

Version     : v1.0, 15:43:59 Jun 24 2021
Status      : float32
[Convert Tools Info]: optional arguments:
	-h    help            show this help message and exit
	-f    input type      path to input float32 tmfile
	-p    input structure path to the network structure of input model(*.prototxt, *.symbol, *.cfg, *.pdmodel)
	-m    input params    path to the network params of input model(*.caffemodel, *.params, *.weight, *.pb, *.onnx, *.tflite, *.pdiparams)
	-o    output model    path to output fp32 tmfile

[Convert Tools Info]: example arguments:
	./convert_tool -f caffe -p ./mobilenet.prototxt -m ./mobilenet.caffemodel -o ./mobilenet.tmfile
```
```bash
$ ./convert_tool/convert_tool -f onnx -m models/yolov5m_leaky_352.onnx -o models/yolov5m_leaky_352.tmfile

---- Tengine Convert Tool ---- 

Version     : v1.0, 15:43:59 Jun 24 2021
Status      : float32
Create tengine model file done: models/yolov5m_leaky_352.tmfile
```
4. Создаём папку с изображениями для калибровки. Желательно иметь 500-1000 изображений из вашего датасета.

5. Запускаем процесс квантизации модели

  Параметры:
```bash
$ ./quant_tool/quant_tool_uint8 -h
[Quant Tools Info]: optional arguments:
	-h    help            show this help message and exit
	-m    input model     path to input float32 tmfile
	-i    image dir       path to calibration images folder
	-f    scale file      path to calibration scale file
	-o    output model    path to output uint8 tmfile
	-a    algorithm       the type of quant algorithm(0:min-max, 1:kl, default is 0)
	-g    size            the size of input image(using the resize the original image,default is 3,224,224)
	-w    mean            value of mean (mean value, default is 104.0,117.0,123.0)
	-s    scale           value of normalize (scale value, default is 1.0,1.0,1.0)
	-b    swapRB          flag which indicates that swap first and last channels in 3-channel image is necessary(0:OFF, 1:ON, default is 1)
	-c    center crop     flag which indicates that center crop process image is necessary(0:OFF, 1:ON, default is 0)
	-y    letter box      the size of letter box process image is necessary([rows, cols], default is [0, 0])
	-k    focus           flag which indicates that focus process image is necessary(maybe using for YOLOv5, 0:OFF, 1:ON, default is 0)
	-t    num thread      count of processing threads(default is 1)

[Quant Tools Info]: example arguments:
	./quant_tool_uint8 -m ./mobilenet_fp32.tmfile -i ./dataset -o ./mobilenet_uint8.tmfile -g 3,224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017

```
```bash
$ ./quant_tool/quant_tool_uint8 -m models/yolov5m_leaky_352.tmfile -i calibration \
-o models/yolov5m_leaky_352_uint8.tmfile -g 3,352,352 -a 0  -w 0,0,0 \
-s 0.003922,0.003922,0.003922 -c 0 -t 4 -b 1 -y 352,352


---- Tengine Post Training Quantization Tool ---- 

Version     : v1.2, 15:15:47 Jun 22 2021
Status      : uint8, per-layer, asymmetric
Input model : models/yolov5m_leaky_352.tmfile
Output model: models/yolov5m_leaky_352_uint8.tmfile
Calib images: calibration
Scale file  : NULL
Algorithm   : MIN MAX
Dims        : 3 352 352
Mean        : 0.000 0.000 0.000
Scale       : 0.004 0.004 0.004
BGR2RGB     : ON
Center crop : OFF
Letter box  : 352 352
YOLOv5 focus: OFF
Thread num  : 4

[Quant Tools Info]: Step 0, load FP32 tmfile.
[Quant Tools Info]: Step 0, load FP32 tmfile done.
[Quant Tools Info]: Step 0, load calibration image files.
[Quant Tools Info]: Step 0, load calibration image files done, image num is 500.
[Quant Tools Info]: Step 1, find original calibration table.
[Quant Tools Info]: Step 1, images 00500 / 00500
[Quant Tools Info]: Step 1, find original calibration table done, output ./table_minmax.scale
[Quant Tools Info]: Thread 4, image nums 500, total time 632535.31 ms, avg time 1265.07 ms
[Quant Tools Info]: Calibration file is using table_minmax.scale
[Quant Tools Info]: Step 3, load FP32 tmfile once again
[Quant Tools Info]: Step 3, load FP32 tmfile once again done.
[Quant Tools Info]: Step 3, load calibration table file table_minmax.scale.
[Quant Tools Info]: Step 4, optimize the calibration table.
[Quant Tools Info]: Step 4, quantize activation tensor done.
[Quant Tools Info]: Step 5, quantize weight tensor done.
[Quant Tools Info]: Step 6, save Int8 tmfile done, models/yolov5m_leaky_352_uint8.tmfile

---- Tengine Int8 tmfile create success, best wish for your INT8 inference has a low accuracy loss...\(^0^)/ ----

```
## Запуск приложения YOLOv5 на Khadas
1. Клонируем репозиторий с приложением
```
git clone https://github.com/SashaAlderson/khadas_yolov5
cd khadas_yolov5
```
2. Устанавливаем OpenCV
```
sudo apt install libopencv-dev
```
3. Собираем приложение
```
chmod u+x build-cv4.sh 
./build-cv4.sh
```
4. Запускаем инференс [модели](https://drive.google.com/drive/folders/1wlErIkcGLRwXylHBNNuMXS2gnXjkixCY?usp=sharing) на нашем изображении
```
$ cd cv4_output
$ ./yolov5 -m yolov5m_leaky_352_uint8.tmfile -i dog.jpg -r 10 -t 4 -s 352

tengine-lite library version: 1.4-dev
authorise ok
input data:0.0039220
:44:44:0:0
Repeat 10 times, thread 4, avg time 30.76 ms, max_time 31.14 ms, min_time 30.41 ms
--------------------------------------
num_detections,43
16: 87%
left = 129,top = 221, width = 0.2, height = 0.5
7: 60%
left = 466,top = 76, width = 0.3, height = 0.2
1: 88%
left = 119,top = 119, width = 0.6, height = 0.5
```
![detection](https://user-images.githubusercontent.com/84590713/162187388-c371b63c-27ec-4fdd-a18f-644d1e368dba.jpg)

## Бенчмарк YOLOv5
[Наша модель](https://drive.google.com/drive/folders/1wlErIkcGLRwXylHBNNuMXS2gnXjkixCY?usp=sharing) обучалась с нуля в течение 600 эпох, чтобы приблизиться к результатам оригинальной YOLOv5m
|        модель        | разрешение |mAP @<br>IoU=0.5:0.95<br>fp32  | FPS* на Khadas VIM3| 
| :------------------: | :--------: |:----------------------------: |  :---------------: | 
|   YOLOv5m-leaky-600  |    640     |           0.440               |       11.5         | 
|   YOLOv5m-leaky-300  |    640     |           0.435               |       11.5         |     
|   YOLOv5m-leaky-600  |    352     |           0.384               |       32.5         |
|   YOLOv5m-leaky-300  |    352     |           0.366               |       32.5         | 
|   YOLOv5m**          |    640     |           0.454               |       5.75         |                          
|   YOLOv5m**          |    352     |           0.392               |       11.4         | 

\*  - без учёта постобработки

\** - оригинальная модель YOLOv5 с SiLU
## Выводы
Замена SiLU на LeakyReLu позволяет увеличить производительность NPU более чем в два раза, но требует больше времени для обучения. За счёт этого получилось достигнуть 32 FPS на разрешении 352х352, сохранив при этом хорошую точность модели.

