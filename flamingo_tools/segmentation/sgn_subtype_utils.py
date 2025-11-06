import os

CUSTOM_THRESHOLDS = {
    "M_LR_000098_L": {"Ntng1": {
        "0492-0500-0581": {"manual": 2, "annotations": 1.41},
        "0561-0179-0502": {"manual": 2.05, "annotations": 1.33},
        "0604-0841-0296": {"manual": 2.32, "annotations": 2.32},
        "0901-0898-0720": {"manual": 1.82, "annotations": 1.82},
        "0929-0749-0290": {"manual": 2.78, "annotations": 2.78},
        "0956-0629-0576": {"manual": 3.36, "annotations": 3.36},
    }},
    "M_LR_000099_L": {"Lypd1": 0.65},
    "M_LR_000184_L": {"Prph": 1},
    "M_LR_000184_R": {"Prph": {
        "0549-0793-0901": {"manual": 1.57, "annotations": 1.57},
        "0618-0900-0594": {"manual": 1.71, "annotations": 1.71},
        "0786-0525-0928": {"manual": 1.48, "annotations": 1.48},
        "0870-0230-0729": {"manual": 1.35, "annotations": 1.08},
        "0911-1142-0849": {"manual": 1.2, "annotations": 1.05},
        "0970-0883-0666": {"manual": 1.83, "annotations": 1.83},
    }},
    "M_LR_000260_L": {"Prph": 0.7},
    "M_LR_N110_L": {"Ntng1": {
        "0592-0600-0527": {"manual": 1.71, "annotations": 1.71},
        "0594-0611-0870": {"manual": 1.9, "annotations": 1.64},
        "0653-1124-0730": {"manual": 1.9, "annotations": 1.56},
        "0840-0887-0878": {"manual": 1.87, "annotations": 1.87},
        "0966-0560-0452": {"manual": 2, "annotations": 1.71},
        "1213-0301-0584": {"manual": 2.33, "annotations": 2.33},
    }},
    "M_LR_N152_L": {"Ntng1": {
        "0519-0929-0610": {"manual": 2, "annotations": 1.95},
        "0752-0997-0849": {"manual": 2, "annotations": 3.51},
        "0791-1331-0750": {"manual": 2, "annotations": 1.15},
        "0794-0904-0420": {"manual": 2, "annotations": 1.66},
        "1013-0330-0643": {"manual": 2, "annotations": 1.45},
        "1013-0613-0479": {"manual": 2, "annotations": 3.83},

    }},
    "M_AMD_N180_L": {"Lypd1": {
        "0441-0660-0521": {"manual": 200, "annotations": 186},
        "0510-0660-0850": {"manual": 179, "annotations": 179.0},
        "0578-1095-0560": {"manual": 200, "annotations": 168.0},
        "0728-0809-0463": {"manual": 220, "annotations": 192.0},
    }},
}


STAIN_TO_TYPE = {
    # Combinations of Calb1 and CR:
    "CR+/Calb1+": "Type Ib",
    "CR-/Calb1+": "Type IbIc",  # Calb1 is expressed at Ic less than Lypd1 but more then CR
    "CR+/Calb1-": "Type Ia",
    "CR-/Calb1-": "Type II",

    # Combinations of Calb1 and Lypd1:
    "Calb1+/Lypd1+": "Type IbIc",
    "Calb1+/Lypd1-": "Type Ib",
    "Calb1-/Lypd1+": "Type Ic",
    "Calb1-/Lypd1-": "inconclusive",  # Can be Type Ia or Type II

    # Combinations of Prph and Tuj1:
    "Prph+/Tuj1+": "Type II",
    "Prph+/Tuj1-": "Type II",
    "Prph-/Tuj1+": "Type I",
    "Prph-/Tuj1-": "inconclusive",

    # Prph is isolated.
    "Prph+": "Type II",
    "Prph-": "Type I",

    # Combinations of CR and Ntng1
    "CR+/Ntng1+": "Type Ib",
    "CR+/Ntng1-": "Type Ia",
    "CR-/Ntng1+": "Type Ic",
    "CR-/Ntng1-": "Type II",

    # Combinations of CR and Lypd1
    "CR+/Lypd1+": "Type Ib",
    "CR+/Lypd1-": "Type Ia",
    "CR-/Lypd1+": "Type Ic",
    "CR-/Lypd1-": "Type II",

    # Combinations of Calb1 and Ntng1
    "Calb1+/Ntng1+": "Type Ib",
    "Calb1+/Ntng1-": "Type Ia",
    "Calb1-/Ntng1+": "Type Ic",
    "Calb1-/Ntng1-": "inconclusive",

}
