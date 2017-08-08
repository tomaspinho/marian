#include "best_hyps.h"

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
                        Scorer& scorer,
                        const Words& filterIndices,
                        Beams &beams,
                        const BeamSize &beamSizes)
{
  BEGIN_TIMER("CalcBeam");

  using namespace mblas;

  mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorer.GetProbs());

  HostVector<float> vCosts;
  for (auto& h : prevHyps) {
    vCosts.push_back(h->GetCost());
  }
  mblas::copy(vCosts.begin(), vCosts.end(), Costs.begin());

  const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

  BroadcastVecColumn(_1 + _2, Probs, Costs);

  if (forbidUNK_) {
    DisAllowUNK(Probs);
  }

  //size_t beamSizeSum = bs.GetTotal();
  size_t beamSizeSum = beamSizes.GetTotal();

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;

  FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst);

  std::vector<HostVector<float>> breakDowns;
  if (returnNBestList_) {
      breakDowns.push_back(bestCosts);
  }

  std::map<size_t, size_t> batchMap;
  //cerr << "beamSizes.size()=" << beamSizes.size() << " batchMap=";
  for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
    uint startPos = beamSizes.Get(batchID).startInd;
    for (size_t t = 0; t < beamSizes.Get(batchID).size; ++t) {
      //cerr << startPos << "=" << batchID << " ";
      batchMap[startPos++] = batchID;
    }
  }
  //cerr << endl;

  for (size_t i = 0; i < beamSizeSum; i++) {
    size_t wordIndex = bestKeys[i] % Probs.dim(1);
    if (isInputFiltered_) {
      wordIndex = filterIndices[wordIndex];
    }

    size_t hypIndex  = bestKeys[i] / Probs.dim(1);
    //std::cerr << "hypIndex=" << hypIndex << std::endl;
    float cost = bestCosts[i];

    HypothesisPtr prevHyp = prevHyps.at(hypIndex);
    HypothesisPtr hyp;
    if (returnAttentionWeights_) {
      hyp.reset(new Hypothesis(prevHyp, wordIndex, hypIndex, cost,
                               GetAlignments(scorer, hypIndex)));
    } else {
      hyp.reset(new Hypothesis(prevHyp, wordIndex, hypIndex, cost));
    }

    if(returnNBestList_) {
      hyp->GetCostBreakdown().resize(1);
      float sum = 0;

      hyp->GetCostBreakdown()[0] = breakDowns[0][i];

      hyp->GetCostBreakdown()[0] -= sum;
    }

    //std::cerr << "i=" << i << " batchMap=" << batchMap[i] << std::endl;

    beams.Add(batchMap[i], hyp);

  }

  PAUSE_TIMER("CalcBeam");

}

std::vector<SoftAlignmentPtr> BestHyps::GetAlignments(Scorer& scorer, size_t hypIndex)
{
  std::vector<SoftAlignmentPtr> alignments;

  if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(&scorer)) {
    const mblas::Matrix &attention = encdec->GetAttention();
    size_t attLength = attention.dim(1);

    SoftAlignment *softAlignment = new SoftAlignment(attLength);
    mblas::copy(
        attention.data() + hypIndex * attLength,
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

std::vector<SoftAlignmentPtr> BestHyps::GetAlignments(const std::vector<ScorerPtr>& scorers,
                                            size_t hypIndex) {
  std::vector<SoftAlignmentPtr> alignments;
  for (auto& scorer : scorers) {
    if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(scorer.get())) {
      const mblas::Matrix &attention = encdec->GetAttention();
      size_t attLength = attention.dim(1);

      SoftAlignment *softAlignment = new SoftAlignment(attLength);
      mblas::copy(
          attention.data() + hypIndex * attLength,
          attLength,
          thrust::raw_pointer_cast(softAlignment->data()),
          cudaMemcpyDeviceToHost
      );

      alignments.emplace_back(softAlignment);
    } else {
      amunmt_UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
    }
  }
  return alignments;
}

}
}


