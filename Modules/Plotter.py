import ROOT
import logging
from array import array
import sys, re
import os
import matplotlib.pyplot as plt
import matplotlib

import numpy as np



def plotROC(tpr, fpr, auc, name):
    c2 = ROOT.TCanvas( 'c2', '', 600, 600 )
    ROOT.gPad.SetLeftMargin(0.13)
    ROOT.gPad.SetRightMargin(0.05)
    c2.SetTickx()
    c2.SetTicky()
    c2.SetGridx();
    c2.SetGridy();
    
    MultiGraph  = ROOT.TMultiGraph()
    
    graph_QPT = ROOT.TGraph( len(fpr), tpr,  1 - fpr )
    graph_QPT.SetLineColor( ROOT.kBlue )
    graph_QPT.SetLineWidth( 3 )
    graph_QPT.SetMarkerStyle( 21 )
    MultiGraph.Add(graph_QPT,"A")
    
    MultiGraph.Draw("A")
    MultiGraph.GetXaxis().SetTitle( 'Signal efficiency' )
    MultiGraph.GetYaxis().SetTitle( 'Background rejection' )
    MultiGraph.GetXaxis().SetTitleSize(0.04)
    MultiGraph.GetXaxis().SetLabelSize(0.04)
    MultiGraph.GetXaxis().SetTitleOffset(1.1)
    MultiGraph.GetYaxis().SetTitleSize(0.04)
    MultiGraph.GetYaxis().SetLabelSize(0.04)
    MultiGraph.GetYaxis().SetTitleOffset(1.5)
    MultiGraph.GetXaxis().SetTickLength(0.018);
    MultiGraph.GetYaxis().SetTickLength(0.018);
    MultiGraph.GetXaxis().SetNdivisions(515)
    MultiGraph.GetYaxis().SetNdivisions(515)
    MultiGraph.GetYaxis().SetRangeUser(0,1.01)
    MultiGraph.GetXaxis().SetRangeUser(0,1.01)
    
    legend = ROOT.TLegend(0.16, 0.15, 0.72, 0.28)
    legend.SetBorderSize(0)
    legend.SetTextAlign(12)
    legend.SetTextFont(42)
    legend.SetTextSize(0.032)
    legend.SetLineColor(0)
    legend.SetLineStyle(0)
    #legend.SetLineWidth(3)
    legend.SetFillColor(0)
    
    legend.AddEntry(graph_QPT, "Q_Transformer (AUC = %0.3f )" % (auc), "l")

    legend.Draw()
    
    c2.Update()
    c2.Modified()
    c2.Update()
    save_path = "./plot/ROCs/"+ name +".pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # mkdir floder
    c2.Print(save_path)