# Демонстрация по предмету «статистическая физика»

## Руководство пользователя
Подробное описание экранов, кнопок и методики проведения экспериментов перенесено в файл [`docs/user_instructions.md`](docs/user_instructions.md). В нём разобраны:

- работа со слайдерами и кнопками панели управления;
- добавление и сопровождение меченых частиц (цветные маркеры и следы);
- интерпретация панели теплового потока и новая шкала на графике;
- сценарий «Сброс измерений → замер потока», когда нужно получить чистые данные.

## Настройка окружения
1. Установите Python 3.11 (проект разрабатывался под эту версию; локально можно использовать более новую, но PyInstaller должен совпадать с целевой версией Python).
2. Для разработки можно использовать привычное `.venv`, а для сборки удобнее завести отдельное окружение `.venv-build`, чтобы в него попадали только обязательные зависимости:
   ```bash
   python3 -m venv .venv-build
   source .venv-build/bin/activate      # macOS/Linux
   # или
   .\.venv-build\Scripts\activate.ps1   # Windows PowerShell
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
   Или, если предпочитаете Pipenv:
   ```bash
   pipenv sync
   ```

## Сборка приложения
PyInstaller использует файл `GradientTemperature.spec`, в котором уже перечислены конфигурация и необходимые ресурсы (`config.json`, `languages/`, `_internal/`).

### macOS (локальная сборка)
```bash
./scripts/build-mac.sh
```
Артефакт появится в `dist/GradientTemperature`. Сценарий ожидает активированное окружение `.venv-build` (или укажите путь к PyInstaller через `PYINSTALLER_BIN`).

### Linux
Сборка должна выполняться в реальной Linux-среде, иначе бинарник может не запуститься из-за несовместимой glibc.
```bash
./scripts/build-linux.sh
```
Если запускаете скрипт из контейнера, примонтируйте проект и убедитесь, что установлены системные зависимости SDL, нужные `pygame`.

### Windows
```powershell
.\scripts\build-windows.ps1
```
Скрипт так же ожидает, что активировано окружение `.venv-build` с установленными зависимостями (`.\.venv-build\Scripts\Activate.ps1`).

### Что получится на выходе
- Папка `dist/GradientTemperature` содержит исполняемый файл и все ресурсы.
- Для распространения удобно упаковать содержимое в архив. На Linux создаётся готовый `dist/GradientTemperature-linux.tar.gz`, на macOS и Windows можно воспользоваться `ditto`/`Compress-Archive` (см. GitHub Actions ниже).

## Как собрать под все платформы с macOS
- PyInstaller не умеет кросс-компилировать, поэтому macOS не может напрямую собрать `.exe` или Linux ELF. Нужны среды с соответствующей ОС: виртуальные машины (Parallels, UTM/VirtualBox), Docker-контейнер для Linux или CI (GitHub Actions, GitLab CI, Azure Pipelines) с матрицей по платформам.
- Для Windows рекомендуемый вариант — GitHub Actions с `runs-on: windows-latest` либо облачный/локальный Windows-хост. Сценарий `scripts/build-windows.ps1` можно запускать прямо в workflow.
- Для Linux удобно использовать контейнер `pyinstaller/pyinstaller:latest` (или любой дистрибутив, поддерживающий нужную glibc). Запустите `./scripts/build-linux.sh` внутри контейнера.
- macOS бинарник собирается напрямую на Mac. Если запускаете PyInstaller в окружении с ограниченными правами, пропишите `export PYINSTALLER_CONFIG_DIR="$PWD/.pyinstaller"` перед сборкой, чтобы кэши создавались в рабочей директории.

## Быстрая проверка
После сборки запустите бинарник на целевой платформе и убедитесь, что доступны все экраны и загружаются ресурсы (`languages/*.json`, изображения из `_internal/images/`). При необходимости обновите файл `GradientTemperature.spec`, чтобы добавить новые ассеты.

## Конфигурация во время исполнения
- В разработке приложение читает `config.json` прямо из корня репозитория.
- В собранном PyInstaller-бандле файл автоматически копируется в:
  - Linux: `~/.config/GradientTemperature/config.json`
  - macOS: `~/Library/Application Support/GradientTemperature/config.json`
  - Windows: `%APPDATA%\GradientTemperature\config.json`
- Этот экземпляр можно редактировать вручную или через меню приложения — изменения сохранятся между запусками.

## Теоретический материал
- Латех-исходники находятся в каталоге `docs/` (`theory_ru.tex`, `theory_en.tex`). Скомпилируйте их в PDF и сохраните под именами `theory_ru.pdf` и `theory_en.pdf`.
- Поместите полученные файлы в каталог `_internal/theory/` рядом с другими ресурсами.
- Экран теории рендерит страницы через библиотеку [`pymupdf`](https://pypi.org/project/pymupdf/). Убедитесь, что пакет установлен в активном окружении (`pip install pymupdf` или `pipenv install pymupdf`).

## Автоматическая сборка (GitHub Actions)
В репозитории добавлен workflow `.github/workflows/build.yml`, который по команде (`workflow_dispatch`) или при пуше тега `v*` собирает архивы для Linux, Windows и macOS. Каждый джоб:

- устанавливает Python 3.12 и зависимости из `requirements.txt`,
- запускает `python -m PyInstaller --noconfirm GradientTemperature.spec`,
- пакует результат в архив (`tar.gz` для Linux, `zip` для Windows/macOS),
- публикует артефакт со имя вида `GradientTemperature-<platform>.{tar.gz|zip}`.

Используйте workflow, чтобы получать повторяемые сборки без ручной настройки окружений.
