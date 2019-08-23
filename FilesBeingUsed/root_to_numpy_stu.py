# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import ROOT
import root_numpy as rnp
import numpy as np

mepz_opts = ['hasTHETA_CHI2', 'hasEPZ', 'hasMEPZ']
poca = "pocas"
PFCatts = []
for mepz in mepz_opts[:1]: 
    print(mepz)
    #If changing this array, make sure that: pT is in index 0, eta is in index 1, and pdgId is the last index
    #calculations in DeepSetTensors_withvalidation depend on these conditions
    if (mepz == 'hasTHETA_CHI2'): PFCatts = ['pt', 'eta', 'phi', 'charge', 'POCA_x', 'POCA_y', 'POCA_z', 'chi2', 'ndof', 'pdgId']
    if (mepz == 'hasEPZ'): PFCatts = ['pt', 'eta', 'phi', 'charge', 'POCA_x', 'POCA_y', 'POCA_z', 'energy', 'pz', 'pdgId']
    if (mepz == 'hasMEPZ'): PFCatts = ['pt', 'eta', 'phi', 'charge', 'POCA_x', 'POCA_y', 'POCA_z', 'mass', 'energy', 'pz', 'pdgId']
    PFCvars = []
    for i in range(len(PFCatts)):
            PFCvars.append(rnp.root2array('VBFH_HToSSTobbbb_MH-125_MS-20_ctauS-10_TuneCUETP8M1_13TeV_Tranche3_PRIVATE-MC_skimmed_decayLength_genVb.root', treename='SignalJets', branches=('PFCandidates.%s'%(PFCatts[i]), 0, 75)))
    PFCfin = np.dstack(PFCvars)
    print(PFCfin[:1])
    np.save('./PFC_data_%s_%s.npy'%(mepz, poca), PFCfin)

    vx = rnp.root2array("VBFH_HToSSTobbbb_MH-125_MS-20_ctauS-10_TuneCUETP8M1_13TeV_Tranche3_PRIVATE-MC_skimmed_decayLength_genVb.root", treename="SignalJets", branches="vx")
    vy = rnp.root2array("VBFH_HToSSTobbbb_MH-125_MS-20_ctauS-10_TuneCUETP8M1_13TeV_Tranche3_PRIVATE-MC_skimmed_decayLength_genVb.root", treename="SignalJets", branches="vy")
    vz = rnp.root2array("VBFH_HToSSTobbbb_MH-125_MS-20_ctauS-10_TuneCUETP8M1_13TeV_Tranche3_PRIVATE-MC_skimmed_decayLength_genVb.root", treename="SignalJets", branches="vz")
    vs = np.transpose(np.vstack([vx, vy, vz]))
    np.save('./SV_true_%s_%s.npy'%(mepz, poca), vs)

    decay_len = rnp.root2array("VBFH_HToSSTobbbb_MH-125_MS-20_ctauS-10_TuneCUETP8M1_13TeV_Tranche3_PRIVATE-MC_skimmed_decayLength_genVb.root", treename="SignalJets", branches="decayLength")
    decay_len = decay_len.reshape(decay_len.shape[0], 1)
    np.save('./decay_len_%s_%s.npy'%(mepz, poca), decay_len)

    print("decaylength:", decay_len.shape, decay_len)
    print("PFpt", PFCvars[0].shape)
    print("PFeta", PFCvars[1].shape)
    print("no. vars ", len(PFCatts))
    print("PFCfin", PFCfin.shape)

    print("vx", vx.shape)
    print("verts", vs.shape)
    print("vx", vx)
    print(vy)
    print(vz)
    print("verts", vs)
