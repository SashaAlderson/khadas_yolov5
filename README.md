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
