#include "best_hyps.h"
#include "common/utils.h"

using namespace std;

namespace amunmt {
namespace GPU {

BestHyps::BestHyps(const God &god)
      : BestHypsBase(
          !god.Get<bool>("allow-unk"),
          god.Get<bool>("n-best"),
          god.Get<std::vector<std::string>>("softmax-filter").size(),
          god.Get<bool>("return-alignment") || god.Get<bool>("return-soft-alignment"),
          god.GetScorerWeights()),
        nthElement_(god.Get<size_t>("beam-size"), god.Get<size_t>("mini-batch")),
        keys(god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
        Costs(god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch"))
{}

void BestHyps::CalcBeam(const Hypotheses& prevHyps,
                        BaseMatrix &probs,
                        const BaseMatrix &attention,
                        const Scorer& scorer,
                        const Words& filterIndices,
                        Beams &beams,
                        const BeamSize &beamSizes)
{
  BEGIN_TIMER("CalcBeam");

  using namespace mblas;

  mblas::Matrix& probsGPU = static_cast<mblas::Matrix&>(probs);

  cerr << "1CalcBeam=" << endl;

  HostVector<float> vCosts;
  for (auto& h : prevHyps) {
    vCosts.push_back(h->GetCost());
  }
  mblas::copy(vCosts.begin(), vCosts.end(), Costs.begin());

  BroadcastVecColumn(_1 + _2, probsGPU, Costs);

  if (forbidUNK_) {
    DisAllowUNK(probsGPU);
  }

  cerr << "2CalcBeam=" << endl;

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;

  nthElement_.getNBestList(beamSizes, probsGPU, bestCosts, bestKeys);
  //cerr << "bestCosts=" << amunmt::Debug(bestCosts, 2) << endl;
  //cerr << "bestKeys=" << amunmt::Debug(bestKeys, 2) << endl;

  cerr << "3CalcBeam=" << endl;

  std::vector<HostVector<float>> breakDowns;
  if (returnNBestList_) {
      breakDowns.push_back(bestCosts);
  }

  cerr << "4CalcBeam=" << endl;

  std::map<size_t, size_t> hypoToBatch;
  size_t tmp = 0;
  for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
    for (size_t t = 0; t < beamSizes.Get(batchID).size; ++t) {
      //cerr << "hypoToBatch=" << tmp << "->" << batchID << endl;
      hypoToBatch[tmp++] = batchID;
    }
  }

  cerr << "5CalcBeam=" << endl;

  size_t beamSizeSum = beamSizes.GetTotal();
  cerr << "beamSizeSum=" << beamSizeSum << endl;

  for (size_t i = 0; i < beamSizeSum; i++) {
    size_t wordIndex = bestKeys[i] % probsGPU.dim(1);
    if (isInputFiltered_) {
      wordIndex = filterIndices[wordIndex];
    }

    size_t hypIndex  = bestKeys[i] / probsGPU.dim(1);
    float cost = bestCosts[i];

    std::cerr << "i=" << i << " hypIndex=" << hypIndex << " " << prevHyps.size() << std::endl;
    HypothesisPtr prevHyp = prevHyps.at(hypIndex);
    HypothesisPtr hyp;
    if (returnAttentionWeights_) {
      hyp.reset(new Hypothesis(prevHyp, wordIndex, hypIndex, cost,
                               GetAlignments(attention, scorer, hypIndex)));
    } else {
      hyp.reset(new Hypothesis(prevHyp, wordIndex, hypIndex, cost));
    }

    if(returnNBestList_) {
      hyp->GetCostBreakdown().resize(1);
      float sum = 0;

      hyp->GetCostBreakdown()[0] = breakDowns[0][i];

      hyp->GetCostBreakdown()[0] -= sum;
    }

    //std::cerr << "i=" << i << " hypoToBatch=" << hypoToBatch[i] << std::endl;

    beams.Add(hypoToBatch[i], hyp);

  }

  PAUSE_TIMER("CalcBeam");

}

std::vector<SoftAlignmentPtr> BestHyps::GetAlignments(const BaseMatrix &attention, const Scorer& scorer, size_t hypIndex)
{
  std::vector<SoftAlignmentPtr> alignments;

  if (const GPU::EncoderDecoder* encdec = dynamic_cast<const GPU::EncoderDecoder*>(&scorer)) {
    const mblas::Matrix &attentionGPU = static_cast<const mblas::Matrix&>(attention);
    size_t attLength = attentionGPU.dim(1);

    SoftAlignment *softAlignment = new SoftAlignment(attLength);
    mblas::copy(
        attentionGPU.data() + hypIndex * attLength,
        attLength,
        thrust::raw_pointer_cast(softAlignment->data()),
        cudaMemcpyDeviceToHost
    );

    alignments.emplace_back(softAlignment);

  } else {
    amunmt_UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
  }

  return alignments;
}

}
}


