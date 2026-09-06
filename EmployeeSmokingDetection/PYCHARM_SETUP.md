# PyCharm-ում տեղադրման ուղեցույց

## 0. Նախապայմաններ

- **Python 3.10 – 3.11** (TensorFlow 2.19-ը 3.12+-ի վրա կարող է խնդիր տալ)
- **PyCharm** Community կամ Professional (երկուսն էլ բավական են)
- **Աշխատող տեսախցիկ**
- ~3 ԳԲ ազատ տեղ (TensorFlow-ը ծանր է)

## 1. Բացիր նախագիծը

Կարևոր՝ PyCharm-ում բացիր **`EmployeeSmokingDetection`** պանակը, ոչ թե repo-ի արմատը։
Հակառակ դեպքում `import config` և `from src import ...` ներմուծումները չեն աշխատի,
և PyCharm-ը ավելորդ ինդեքսավորում կանի .NET ֆայլերի վրա։

```
File → Open… → ընտրիր EmployeeSmokingDetection պանակը → Open
```

## 2. Ստեղծիր virtual environment

```
File → Settings → Project → Python Interpreter
  → ⚙ → Add… → Virtualenv Environment → New
  → Base interpreter: Python 3.11
  → OK
```

macOS-ում՝ `PyCharm → Settings`։

## 3. Տեղադրիր կախվածությունները

PyCharm-ը սովորաբար ինքն է առաջարկում, երբ բացում ես `requirements.txt`-ը։
Կամ ձեռքով, PyCharm-ի Terminal-ում (ներքևի վահանակ)․

```bash
pip install -r requirements.txt
pip install pytest
```

⚠️ **Զգուշացում.** եթե համակարգում արդեն տեղադրված է `opencv-python`,
նախ ջնջիր այն — `opencv-contrib-python`-ի հետ կոնֆլիկտ է տալիս, և միայն
contrib տարբերակն ունի `cv2.face` մոդուլը (դեմքի ճանաչման համար)․

```bash
pip uninstall opencv-python
```

Ստուգիր, որ ամեն ինչ տեղում է․

```bash
python -c "import tensorflow, cv2; print(tensorflow.__version__, cv2.__version__, hasattr(cv2,'face'))"
```

Պետք է տպի երեք արժեք և վերջում `True`։

## 4. Run Configuration-ները

`.idea/runConfigurations/`-ում արդեն պատրաստ են 6 կոնֆիգուրացիա։ Բացելուց հետո
դրանք կհայտնվեն վերևի աջ անկյան բացվող ցանկում․

> ℹ️ **Ուշադրություն.** repo-ի արմատային `.gitignore`-ը անտեսում է `.idea/`-ն, ուստի
> այս կոնֆիգուրացիաները **git-ով չեն գալիս** — դրանք միայն ուղարկված
> `EmployeeSmokingDetection.tar.gz` արխիվի մեջ են։ Եթե նախագիծը վերցրել ես git-ից,
> կա՛մ պատճենիր `.idea/` պանակը արխիվից, կա՛մ ձեռքով ստեղծիր
> կոնֆիգուրացիաները ներքևի աղյուսակի տվյալներով
> (`Run → Edit Configurations… → + → Python`)։

| # | Անուն | Ի՞նչ է անում |
|---|---|---|
| 1 | Train model | Սովորեցնում է դասակարգիչը (երկու փուլով) |
| 2 | Evaluate model | Confusion matrix, ROC, AUC |
| 3 | Register employee | Գրանցում է աշխատակցի դեմքը տեսախցիկից |
| 4 | Detect webcam (live) | Իրական ժամանակի հայտնաբերում |
| 5 | Dashboard (Flask) | Վահանակ՝ http://localhost:5000 |
| 6 | Tests (pytest) | Գործարկում է թեստերը |

Եթե ցանկում չեն երևում՝ `Run → Edit Configurations…` և ստուգիր, որ
module-ի անունը համընկնում է քո նախագծի անվան հետ։

**№3-ի անունը փոխիր** քո աշխատակցի իրական անվան․
`Run → Edit Configurations… → 3. Register employee → Parameters`

## 5. Աշխատանքի հերթականությունը

```
[1] Դիր նկարները՝  data/raw/smoking/  և  data/raw/not_smoking/
         (նվազագույնը մի քանի հարյուր հատ յուրաքանչյուր դասի համար)
              ↓
[2] Run → "1. Train model"        ← կտևի 20-60 րոպե CPU-ի վրա
              ↓
[3] Run → "2. Evaluate model"     ← ստուգիր AUC-ը; <0.85 = պետք է ավելի շատ տվյալ
              ↓
[4] Run → "3. Register employee"  ← կրկնիր յուրաքանչյուր աշխատակցի համար
              ↓
[5] Run → "4. Detect webcam"      ← 'q' սեղմելով դուրս գալ
              ↓
[6] Run → "5. Dashboard"          ← բացիր http://localhost:5000
```

## 6. Հաճախակի խնդիրներ

**`ModuleNotFoundError: No module named 'config'`**
PyCharm-ում բացել ես սխալ պանակը։ Պետք է `EmployeeSmokingDetection`-ը լինի
project root։ Կամ՝ աջ սեղմիր պանակի վրա → `Mark Directory as → Sources Root`։

**`AttributeError: module 'cv2' has no attribute 'face'`**
Տեղադրված է սովորական `opencv-python`։ Տես բաժին 3-ի զգուշացումը։

**`Could not open camera index 0`**
- Այլ ծրագիր (Zoom, Teams) զբաղեցրել է տեսախցիկը — փակիր այն
- macOS՝ System Settings → Privacy & Security → Camera → թույլատրիր PyCharm-ին
- Փորձիր այլ ինդեքս՝ `config.py`-ում `CAMERA_INDEX = 1`

**`FileNotFoundError: No trained model found at models/smoking_classifier.keras`**
Դեռ չես սովորեցրել մոդելը։ Նախ գործարկիր «1. Train model»-ը։

**Ուսուցումը շատ դանդաղ է**
Նորմալ է CPU-ի վրա։ Փորձի համար իջեցրու՝ `--epochs 3 --fine-tune-epochs 2`։

**Տեսախցիկի պատուհանը չի փակվում**
Սեղմիր `q` հենց **պատուհանի** վրա (ոչ թե կոնսոլում)։

## 7. Կարգավորումներ

Բոլոր շեմերը՝ `config.py`-ում։ Ամենակարևորները․

```python
SMOKING_CONFIDENCE_THRESHOLD = 0.65        # ↑ = ավելի քիչ կեղծ ազդանշան
CONSECUTIVE_FRAMES_TO_CONFIRM = 5          # ↑ = ավելի վստահ, բայց դանդաղ
FACE_RECOGNITION_CONFIDENCE_THRESHOLD = 70 # ↓ = ավելի խիստ ինքնության համընկնում
CAMERA_INDEX = 0                            # տեսախցիկի համարը
```
