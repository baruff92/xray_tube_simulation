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
#include <random>
#include <string>
#include <sstream>
#include <TPaveStats.h>
#include <TStyle.h>

void put_noise()
{
  std::cout << "We simulate a Si detector with noise" << std::endl;
  TCanvas *c1 = new TCanvas("c1","c1");


  // Create a vector with the random timeshifts for each particle
  random_device rnd_device;
  mt19937 mersenne_engine {rnd_device()};

  std:vector<double> energy_resolution = {0.108,0.144, 0.180, 0.216}; // eV
  std::vector<int> colors = {1,2,3,4,6};

  int i = 0;

  for (auto& det_resolution : energy_resolution)
  {
    std::ostringstream text_1;
    text_1 << "sigma: " << det_resolution << " eV (" << det_resolution/3.6*1000 << " e-)";

    TH1F* gaussian_histogram = new TH1F(text_1.str().c_str(), "gaussian_histogram;random number (keV);Counts", 200, -0.5, 0.5);
    gaussian_histogram->SetLineColor(colors[i]);

    std::normal_distribution<> dist(0, det_resolution);
    std::cout << "The detector energy resolution is: "<< det_resolution;
    for(int i=0; i< 100000; i++)
    {
      double rand_numb = dist(mersenne_engine);
      gaussian_histogram->Fill(rand_numb);
    }
    if (i==0) gaussian_histogram->Draw();
    else gaussian_histogram->Draw("Sames");
    i++;
  }
}
