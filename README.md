# Установка Ubuntu на Khadas VIM3
1. Следуя [инструкциям](https://docs.khadas.com/linux/vim3/index.html) из раздела Quick start устанавливаем Ubuntu. Образ можно взять с [официального сайта](https://docs.khadas.com/linux/firmware/Vim3UbuntuFirmware.html) или брать уже 
протестированную версию [V1.0.7-210625](https://drive.google.com/drive/folders/1FUXloO80ecwliYHQxfVgwSeL5ceh2Hhl?usp=sharing).
# YOLOv5
1. Клонируем форк репозитория, с leakyReLU свёртками и экспортом в ONNX обрезанного слоя Detect
```
git clone https://github.com/SashaAlderson/yolov5
```
2. Обучаем сеть в течение 300 эпох следуя гайдам, либо используем [предобученную](https://drive.google.com/drive/folders/1wlErIkcGLRwXylHBNNuMXS2gnXjkixCY?usp=sharing).
# Конвертация на NPU
Клонируем репозиторий с готовыми бинарными файлами для конвертации
```
git clone https://github.com/SashaAlderson/tengine_khadas_sdk
cd tengine_khadas_sdk/tengine_tools
```
Устанавливаем opencv
```
sudo apt install libopencv-dev
```
Конвертируем модель в tmfile
```
$ ./convert_tool/convert_tool -f onnx -m models/yolov5m_leaky_352.onnx -o models/yolov5m_leaky_352.tmfile
```
```
---- Tengine Convert Tool ---- 

Version     : v1.0, 15:43:59 Jun 24 2021
Status      : float32
Create tengine model file done: models/yolov5m_leaky_352.tmfile
```
Создаём папку с изображениями для калибровки. Желательно иметь 500-1000 изображений из вашего датасета.

Запускаем процесс квантизации модели
```
$ ./quant_tool/quant_tool_uint8 -m models/yolov5m_leaky_352.tmfile -i calibration/ \
-o models/yolov5m_leaky_352_uint8.tmfile -g 3,352,352 -a 0  -w 0,0,0 \
-s 0.003922,0.003922,0.003922 -c 0 -t 4 -b 1 -y 352,352


---- Tengine Post Training Quantization Tool ---- 

Version     : v1.2, 15:15:47 Jun 22 2021
Status      : uint8, per-layer, asymmetric
Input model : models/yolov5m_leaky_352.tmfile
Output model: models/yolov5m_leaky_352_uint8.tmfile
Calib images: calibration/
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
