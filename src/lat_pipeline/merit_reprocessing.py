import numpy as np

def merit_reprocessing(merit_vars: dict) -> dict:
    """
    Processes a single event's merit variables using safe, pure Python math.
    Strictly aligned with the 17 output variables from the original script.
    """
    df1 = {}

    # Helper to safely grab variables (defaults to NaN if missing)
    def get_val(key):
        return float(merit_vars.get(key, np.nan))

    # Grab required input variables
    CalNewCfpSelChiSq = get_val("CalNewCfpSelChiSq")
    CalELayer1 = get_val("CalELayer1")
    CalELayer0 = get_val("CalELayer0")
    EvtJointEnergy = get_val("EvtJointEnergy")
    TkrEnergyCorr = get_val("TkrEnergyCorr")
    Tkr1ZDir = get_val("Tkr1ZDir")
    Tkr1ToTTrAve = get_val("Tkr1ToTTrAve")
    CalXEcntr = get_val("CalXEcntr")
    CalYEcntr = get_val("CalYEcntr")
    Tkr1Z0 = get_val("Tkr1Z0")
    Tkr1X0 = get_val("Tkr1X0")
    Tkr1Y0 = get_val("Tkr1Y0")
    EvtCalCsIRLn = get_val("EvtCalCsIRLn")
    CalZEcntr = get_val("CalZEcntr")
    Tkr1CoreHC = get_val("Tkr1CoreHC")
    Tkr1Hits = get_val("Tkr1Hits")
    TkrNumStripsThin = get_val("TkrNumStripsThin")
    TkrNumStripsThick = get_val("TkrNumStripsThick")
    TkrNumStripsBlank = get_val("TkrNumStripsBlank")
    TkrTree1CalDoca68 = get_val("TkrTree1CalDoca68")
    CalTransRms = get_val("CalTransRms")

    # 1. CalNewCfpSelChiSqLog
    if CalNewCfpSelChiSq > 0:
        df1["CalNewCfpSelChiSqLog"] = np.log10(CalNewCfpSelChiSq)
    else:
        df1["CalNewCfpSelChiSqLog"] = np.nan

    # 2. CalELayer10RatioES
    if CalELayer0 > 0 and EvtJointEnergy > 0:
        logE = np.log10(EvtJointEnergy)
        denominator = -0.31 * (logE**2) + 4.196 * logE - 10.5
        if denominator != 0:
            df1["CalELayer10RatioES"] = (CalELayer1 / CalELayer0) / denominator
        else:
            df1["CalELayer10RatioES"] = np.nan
    else:
        df1["CalELayer10RatioES"] = np.nan

    # 3. TkrEnergyFracLogZS
    if EvtJointEnergy > 0 and TkrEnergyCorr > 0:
        df1["TkrEnergyFracLogZS"] = np.log10(TkrEnergyCorr / max(1.0, EvtJointEnergy)) - 0.8 * (Tkr1ZDir + 1)
    else:
        df1["TkrEnergyFracLogZS"] = np.nan

    # 4. Tkr1ToTTrAve
    df1["Tkr1ToTTrAve"] = Tkr1ToTTrAve

    # 5. CalEdgeDist (and saving centroids directly)
    df1["CalEdgeDist"] = max(abs(CalXEcntr), abs(CalYEcntr))
    
    # 6. CalXEcntr
    df1["CalXEcntr"] = CalXEcntr
    
    # 7. CalYEcntr
    df1["CalYEcntr"] = CalYEcntr

    # 8. Tkr1Z0
    df1["Tkr1Z0"] = Tkr1Z0

    # 9. TkrEdgeDist
    df1["TkrEdgeDist"] = max(abs(Tkr1X0), abs(Tkr1Y0))

    # 10. EvtCalCsIRLn
    df1["EvtCalCsIRLn"] = EvtCalCsIRLn

    # 11. CalZEcntr
    df1["CalZEcntr"] = CalZEcntr

    # 12. Tkr1CoreHCRatio
    if Tkr1Hits > 0:
        df1["Tkr1CoreHCRatio"] = Tkr1CoreHC / max(1.0, Tkr1Hits)
    else:
        df1["Tkr1CoreHCRatio"] = np.nan

    # 13. TkrNumStripsThinLog
    if TkrNumStripsThin > 0:
        df1["TkrNumStripsThinLog"] = np.log10(TkrNumStripsThin)
    else:
        df1["TkrNumStripsThinLog"] = np.nan

    # 14. TkrNumStripsThickLog
    if TkrNumStripsThick > 0:
        df1["TkrNumStripsThickLog"] = np.log10(TkrNumStripsThick)
    else:
        df1["TkrNumStripsThickLog"] = np.nan

    # 15. TkrNumStripsBlankLog
    if TkrNumStripsBlank > 0:
        df1["TkrNumStripsBlankLog"] = np.log10(TkrNumStripsBlank)
    else:
        df1["TkrNumStripsBlankLog"] = np.nan

    # 16. TkrTree1CalDoca68Log
    if TkrTree1CalDoca68 > 0:
        df1["TkrTree1CalDoca68Log"] = np.log10(TkrTree1CalDoca68)
    else:
        df1["TkrTree1CalDoca68Log"] = np.nan

    # 17. CalTransRms
    df1["CalTransRms"] = CalTransRms

    # Replace all physical NaNs with 0.0 so PyTorch dataloaders don't crash
    for key, value in df1.items():
        if np.isnan(value):
            df1[key] = 0.0

    return df1