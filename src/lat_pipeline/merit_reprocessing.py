import numpy as np

def merit_reprocessing(merit_vars: dict) -> dict:
    """
    Processes a single event's merit variables using safe, pure Python math.
    No Pandas overhead, natively handles single scalar values.
    """
    df1 = {}

    # Helper to safely grab variables (defaults to NaN if missing)
    def get_val(key):
        return float(merit_vars.get(key, np.nan))

    # Grab everything we need for a single event
    CalNewCfpSelChiSq = get_val("CalNewCfpSelChiSq")
    CalELayer1 = get_val("CalELayer1")
    CalELayer0 = get_val("CalELayer0")
    EvtJointEnergy = get_val("EvtJointEnergy")
    TkrEnergyCorr = get_val("TkrEnergyCorr")
    Tkr1ZDir = get_val("Tkr1ZDir")
    CalXEcntr = get_val("CalXEcntr")
    CalYEcntr = get_val("CalYEcntr")
    Tkr1X0 = get_val("Tkr1X0")
    Tkr1Y0 = get_val("Tkr1Y0")
    Tkr1Hits = get_val("Tkr1Hits")
    Tkr1CoreHC = get_val("Tkr1CoreHC")

    # CalNewCfpSelChiSqLog
    if CalNewCfpSelChiSq > 0:
        df1["CalNewCfpSelChiSqLog"] = np.log10(CalNewCfpSelChiSq)
    else:
        df1["CalNewCfpSelChiSqLog"] = np.nan

    # CalELayer10RatioES
    if CalELayer0 > 0 and EvtJointEnergy > 0:
        logE = np.log10(EvtJointEnergy)
        denominator = -0.31 * (logE**2) + 4.196 * logE - 10.5
        if denominator != 0:
            df1["CalELayer10RatioES"] = (CalELayer1 / CalELayer0) / denominator
        else:
            df1["CalELayer10RatioES"] = np.nan
    else:
        df1["CalELayer10RatioES"] = np.nan

    # TkrEnergyFracLogZS
    if EvtJointEnergy > 0 and TkrEnergyCorr > 0:
        # np.log10 requires > 0, max(1.0, E) protects the denominator
        df1["TkrEnergyFracLogZS"] = np.log10(TkrEnergyCorr / max(1.0, EvtJointEnergy)) - 0.8 * (Tkr1ZDir + 1)
    else:
        df1["TkrEnergyFracLogZS"] = np.nan

    # Tkr1ToTTrAve
    df1["Tkr1ToTTrAve"] = get_val("Tkr1ToTTrAve")

    # CalEdgeDist & Centroids
    df1["CalEdgeDist"] = max(abs(CalXEcntr), abs(CalYEcntr))
    df1["CalXEcntr"] = CalXEcntr
    df1["CalYEcntr"] = CalYEcntr

    # Tkr1Z0
    df1["Tkr1Z0"] = get_val("Tkr1Z0")

    # TkrEdgeDist
    df1["TkrEdgeDist"] = max(abs(Tkr1X0), abs(Tkr1Y0))

    # EvtCalCsIRLn & CalZEcntr
    df1["EvtCalCsIRLn"] = get_val("EvtCalCsIRLn")
    df1["CalZEcntr"] = get_val("CalZEcntr")

    # Tkr1CoreHCRatio
    if Tkr1Hits > 0:
        df1["Tkr1CoreHCRatio"] = Tkr1CoreHC / Tkr1Hits
    else:
        df1["Tkr1CoreHCRatio"] = 0.0

    return df1