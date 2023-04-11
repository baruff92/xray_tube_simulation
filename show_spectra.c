#include <TCanvas.h>
#include <TFile.h>
#include <TTree.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string>
#include <sstream>

void plot_alu_spectra()
{
  gStyle->SetOptStat(0);
  std::cout << "Plotting the spectra" << std::endl;
  TCanvas *c4 = new TCanvas("c4","c4");

  std::vector<std::string> root_files = {
                                        "Al_0um.root",
                                        "Al_18um.root",
                                        "Al_36um.root",
                                        "Al_54um.root",
                                        "Al_72um.root",
                                        };

  int i=0;
  std::vector<int> colors = {1,2,3,4,6};
  std::vector<double> thickness;
  std::vector<double> intensity;
  for (auto f : root_files)
  {
    std::cout << f << std::endl;
    TFile *fileSp = new TFile(f.c_str());
    TH1D *spectr = new TH1D();
    spectr = (TH1D*)fileSp->Get("energySpectrumFluenceTrack");
    std::cout << "  Entries:" << spectr->GetEntries() << std::endl;
    spectr->SetTitle("");
    spectr->GetXaxis()->Set(spectr->GetNbinsX(), 0, 70);
    spectr->GetXaxis()->SetTitle("Energy (keV)");
    spectr->SetLineColor(colors[i]);
    spectr->Draw("hist Sames");

    string start,end;
    start = f.substr(1,f.find("_")+1);
    end = f.substr(f.find("um")+2,f.size());
    std::cout<< start << " " << end << std::endl;

    TLatex *text = new TLatex();
    text->SetTextSize(0.03);
    text->SetTextFont(42);
    text->SetTextColor(colors[i]);
    text->DrawTextNDC(0.45, 0.7-i*0.08, f.c_str());
    std::ostringstream text_1;
    text_1 << "Entries: " << spectr->GetEntries();
    text->DrawTextNDC(0.6, 0.7-i*0.08, text_1.str().c_str());
    // delete spectr;
    // delete fileSp;
    i++;
  }

  // TFile *fileSp = new TFile("Al_0um.root");
  // TH1D *spectr = new TH1D();
  // spectr = (TH1D*)fileSp->Get("energySpectrumFluenceTrack");
  //
  // TCanvas *c5 = new TCanvas("c5","c5");
  // spectr->Draw();

  // spectr->Draw("sames");

}
