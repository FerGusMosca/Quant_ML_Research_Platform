class ArgyInstrumentClassifier:
    """
    Classifies Argentine fixed-income instruments (bonds, bills, etc.)
    according to official BYMA/BCRA conventions and Excel mapping.
    """

    @staticmethod
    def classify(desc: str):
        """
        Returns a dictionary:
        {
            "category": <Clase>,      # same as Excel col A
            "method": <Tipo de cálculo de TIR correcto>  # same as Excel col E
        }
        """
        d = desc.upper().strip()

        # --- Bonos soberanos USD ---
        if any(x in d for x in [" AL30D", " GD35D", " GD30D", " AE38D", " GD46D"]):
            return {
                "category": "Bonos soberanos USD (Ley NY)",
                "method": "YTM nominal en USD"
            }

        if any(x in d for x in [" AL30C", " GD30C", " AE38C", " AL30 ", " AE38 "]):
            return {
                "category": "Bonos soberanos USD (Ley AR)",
                "method": "YTM nominal en USD"
            }

        # --- Bonos CER / Ajustables por inflación ---
        if any(x in d for x in ["DICP", "TX26", "TX28", "TX31", "CER"]):
            return {
                "category": "Bonos ajustables por CER (Bonos CER)",
                "method": "YTM real ajustado por CER"
            }

        # --- Bonos dollar-linked ---
        if any(x in d for x in ["TV24", "TV25", "TV26", "TVD", "TVP", "TVPY"]):
            return {
                "category": "Bonos dollar-linked",
                "method": "YTM nominal con ajuste FX"
            }

        # --- Bonos duales (inflación o FX) ---
        if any(x in d for x in ["T2V", "T3V"]):
            return {
                "category": "Bonos duales (inflación o FX)",
                "method": "YTM mixto (mayor entre infl. y FX)"
            }

        # --- Bonos Badlar (tasa variable) ---
        if any(x in d for x in ["TB21", "TB27", "BADLAR"]):
            return {
                "category": "Bonos Badlar (tasas variables)",
                "method": "YTM con tasa flotante esperada"
            }

        # --- Bonos duales BADLAR-CER ---
        if any(x in d for x in ["TDF", "TDF24", "TDF25"]):
            return {
                "category": "Bonos duales BADLAR-CER",
                "method": "YTM mixto"
            }

        # --- Letras del Tesoro ---
        if "LECAP" in d or "LETE" in d or "J26" in d:
            return {
                "category": "Letras del Tesoro",
                "method": "TNA simple o YTM cero cupón"
            }

        # --- BOPREAL ---
        if "BOPREAL" in d:
            return {
                "category": "BOPREAL",
                "method": "YTM nominal en USD"
            }

        # --- Bonos provinciales ---
        if any(x in d for x in ["PBA", "CABA", "NEUQ", "SF", "BAA"]) and not "BOPREAL" in d:
            return {
                "category": "Bonos provinciales",
                "method": "YTM nominal"
            }

        # --- Bonos corporativos ---
        if any(x in d for x in ["YPFD", "TGSG", "IRSA", "TGSU", "BACS", "BGBD", "BGRD", "BFC", "AY24"]):
            if "CER" in d:
                return {
                    "category": "Bonos corporativos CER",
                    "method": "YTM real ajustado por CER"
                }
            return {
                "category": "Bonos corporativos",
                "method": "YTM nominal"
            }

        # --- Letras / Pases / Cauciones ---
        if "LELIQ" in d:
            return {
                "category": "Letras BCRA (LELIQ)",
                "method": "TNA simple"
            }

        if "PASE" in d:
            return {
                "category": "Pases BCRA",
                "method": "TNA simple diaria"
            }

        if "CAUC" in d:
            return {
                "category": "Caución bursátil",
                "method": "TNA simple"
            }

        # --- Default ---
        return {
            "category": "Otro / No identificado",
            "method": "Desconocido"
        }
