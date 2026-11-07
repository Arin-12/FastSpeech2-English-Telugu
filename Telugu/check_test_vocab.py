from pathlib import Path

ns = {}
exec(
    Path("Telugu/FastSpeech2-Telugu/text/symbols.py").read_text(encoding="utf-8"),
    ns
)
symbols = set(ns["symbols"])

tests = [
    "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన మందుల ప్రభావాన్ని విశ్లషే ించారు",
    "తాడుపై అతని పట్టుచాలా బలంగా ఉంది",
    "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది",
    "పులి అడ విలో నిశ్శబ్ద గా నడిచింది",
    "ఈ సమావేశం రెండు వేల ఇరవై ఆరు నవంబర్ పన్నెండవ తేదీ ఉదయం పది గంటల నలభై ఐదు నిమిషాలకు ప్రారంభమవుతుంది సమయానికి సిద్ధంగా ఉంచండి"
]
for i, text in enumerate(tests, 1):
    missing = sorted(set(text) - symbols)
    print(f"\nTest {i}: {'PASS' if not missing else 'MISSING'}")
    for ch in missing:
        print(repr(ch), f"U+{ord(ch):04X}")
