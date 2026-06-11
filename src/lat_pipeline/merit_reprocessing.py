import numpy as np
import pandas as pd


def merit_reprocessing(merit_vars: dict, energy: float) -> dict:
    df0 = pd.DataFrame([merit_vars])
    df1 = pd.DataFrame()
    #CalNewCfpSelChiSqLog
    df1["CalNewCfpSelChiSqLog"] = np.log10(df0["CalNewCfpSelChiSq"].values)

    #CalELayer10RatioES"
    df1["CalELayer10RatioES"] = (df0["CalELayer1"].values)/(df0["CalELayer0"].values) /(-0.31*(np.log10(df0["EvtJointEnergy"].values))**2 + 4.196 * np.log10(df0["EvtJointEnergy"].values) - 10.5)
    #TkrEnergyFracLogZS
    EvtJointEnergy = df0["EvtJointEnergy"].values
    TkrEnergyCorr = df0["TkrEnergyCorr"].values
    TkrEnergyFracLogZS = []
    for i in range(0,len(EvtJointEnergy)):
        TkrEnergyFracLogZS.append(np.log10(TkrEnergyCorr[i]/max(1.,EvtJointEnergy[i])))
    df1["TkrEnergyFracLogZS"] = np.array(TkrEnergyFracLogZS) -0.8*(df0["Tkr1ZDir"].values +1)
    #Tkr1ToTTrAve
    df1["Tkr1ToTTrAve"] = df0["Tkr1ToTTrAve"].values
    #CalEdgeDist
    CalXEcntr = df0["CalXEcntr"].values
    CalYEcntr = df0["CalYEcntr"].values
    CalEdgeDist = []
    for i in range(0,len(CalXEcntr)):
        CalEdgeDist.append(max(abs(CalXEcntr[i]), abs(CalYEcntr[i])))
    df1["CalEdgeDist"] = np.array(CalEdgeDist)
    #CalXEcntr
    df1["CalXEcntr"] = CalXEcntr
    df1["CalYEcntr"] = CalYEcntr
    #Tkr1Z0
    df1["Tkr1Z0"] = df0["Tkr1Z0"].values
    #TkrEdgeDist
    Tkr1X0 = df0["Tkr1X0"].values
    Tkr1Y0 = df0["Tkr1Y0"].values
    TkrEdgeDist = []
    for i in range(0,len(CalXEcntr)):
        TkrEdgeDist.append(max(abs(Tkr1X0[i]), abs(Tkr1Y0[i])))
    df1["TkrEdgeDist"] = np.array(TkrEdgeDist)
    #EvtCalCsIRLn
    df1["EvtCalCsIRLn"] = df0["EvtCalCsIRLn"].values
    #CalZEcntr
    df1["CalZEcntr"] = df0["CalZEcntr"].values
    #Tkr1CoreHCRatio
    Tkr1CoreHC = df0["Tkr1CoreHC"].values
    Tkr1Hits = df0["Tkr1Hits"].values
    Tkr1CoreHCRatio = []
    for i in range(0,len(CalXEcntr)):
        Tkr1CoreHCRatio.append(Tkr1CoreHC[i]/max(1,Tkr1Hits[i]))
    df1["Tkr1CoreHCRatio"] = np.array(Tkr1CoreHCRatio)
    #TkrNumStripsThinLog
    df1["TkrNumStripsThinLog"] = np.log10(df0["TkrNumStripsThin"].values)
    #TkrNumStripsThickLog
    df1["TkrNumStripsThickLog"] = np.log10(df0["TkrNumStripsThick"].values)
    #TkrNumStripsBlankLog
    df1["TkrNumStripsBlankLog"] = np.log10(df0["TkrNumStripsBlank"].values)

    #TkrTree1CalDoca68Log
    df1["TkrTree1CalDoca68Log"] = np.log10(df0["TkrTree1CalDoca68"].values)
    #Cal1TransRms
    df1["CalTransRms"] = df0["CalTransRms"].values

    return df1.to_dict()