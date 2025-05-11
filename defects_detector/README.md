# Patch Cutter
Реализация класса расположена в `defects_detector/patch_cutter.py`. Он нужен для того, 
чтобы нарезать исходные 3d-данные на патчи, которые можно потом использовать для обнаружения дефектов или обучения.

Данный класс использует "интерфейсы" `BaseDataLoader` и `BasePatchCutter`. В `/defects_detector/preprocessing/mvtec.py`
реализованы интерфейсы для работы с данными в формате датасета MVTec3D-AD.

Для проверки работы необходимо установить зависимости из `requirements.txt` (их много...). И нужно предобработать данные 
согласно описанию из `README.md` в корне репозитория, а именно [удалить фон](README.md###Preprocessing)

Для того чтобы нарезать патчи с помощью `patch_cutter.py`, необходимо выполнить команду (из корня репозитория):
```bash
# для данных в формате MVTec3D-AD, но не обязательно со строгим соответствием структуре датасета, то есть ожидается, 
# что в директории есть поддиректория `xyz`, в которой находятся .tiff файлы
python -m defects_detector.preprocessing.patches.cli --datasets_path /mnt/d/Downloads/MVTec3D-AD/cookie/test/combined \
  --save_path /mnt/d/Downloads/MVTec3D-AD/cookie/test/combined --split custom
```
Если хотим нарезать патчи как с использованием `cut_pathches.py` из корня репо, то нужно указать путь к датасету и 
один из флагов `train`, `test`, `validation`, `pretrain`, либо `all`, который нарежет патчи для всех поддиректорий.
```bash
 python -m defects_detector.preprocessing.patches.cli --datasets_path /mnt/d/Downloads/MVTec3D-AD \
  --save_path /mnt/d/Downloads/MVTec3D-AD --split test
```

Кроме того, есть флаги для изменения параметров нарезки патчей (кол-во точек в патче, размер изображения и т.п.), 
но я использую дефолтные, так как и данные дефолтные

## Known issues
Если попытаться после этого стартовать обнаружение дефектов, то может выдать ошибку о превышении лимита открытых 
файловых дескрипторов.
```bash 
ulimit -n 4096
```
4к файлов должно хватить для работы.