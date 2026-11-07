# CarTube

YouTube-ը Kia-ի (կամ ցանկացած մեքենայի) էկրանին՝ **Android Auto**-ի միջոցով, Android հեռախոսից (Redmi և այլն)։

## Ինչպես ստանալ APK-ն
1. GitHub-ում բացեք ռեպոզիտորիայի **Actions** բաժինը → **Build CarTube APK**։
2. Բացեք վերջին կանաչ (✅) build-ը և ներքևում **Artifacts** բաժնից ներբեռնեք **CarTube-apk**-ը։
3. Zip-ից հանեք `app-debug.apk`-ը, ուղարկեք հեռախոսին և տեղադրեք (թույլատրեք «Install unknown apps»)։

## Կարգավորում
1. **Android Auto-ի developer ռեժիմ.** Android Auto-ի կարգավորումներ → **Version**-ի վրա սեղմեք ~10 անգամ → ⋮ → **Developer settings** → նշեք **Unknown sources**։
2. **Redmi/Xiaomi.** Settings → Apps → **CarTube** և **Android Auto** → Battery saver → **No restrictions**, միացրեք **Autostart**։
3. Միացրեք հեռախոսը մեքենային։ Եթե CarTube-ը ցանկում չի երևում, Android Auto → **Customize launcher** → միացրեք այն։

## Օգտագործում
- Մեքենայի էկրանին բացվում է m.youtube.com-ը։ Սեղմելով ընտրում եք, մատով սահեցնելով՝ թերթում։
- Կոճակներ՝ **Back**, **Home**, **Search** (մեքենայի ստեղնաշար), **Reload**։
- Հեռախոսում բացեք CarTube-ը, գրեք որոնում կամ YouTube հղում և սեղմեք **Ուղարկել մեքենային**։

## Ինչպես է աշխատում (տեխնիկական)
Հավելվածը գրանցված է որպես Android Auto-ի «navigation» հավելված (Car App Library), որը ստանում է մեքենայի էկրանի `Surface`-ը։
Այդ surface-ի վրա ստեղծվում է `VirtualDisplay`, որի վրա `Presentation`-ի մեջ ցուցադրվում է `WebView`։
Մեքենայի էկրանի հպումները (`onClick`, `onScroll`, `onFling`, `onScale`) փոխանցվում են WebView-ին։

⚠ Տեսանյութ դիտեք **միայն կայանված ժամանակ**։ Հավելվածն անձնական օգտագործման համար է, Google Play-ում հրապարակելու համար չէ։
