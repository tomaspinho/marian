#include "best_hyps.h"

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
                        std::vector<uint>& beamSizes)
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

  size_t beamSizeSum = std::accumulate(beamSizes.begin(), beamSizes.end(), 0);

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;

  FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst);

  std::vector<HostVector<float>> breakDowns;
  if (returnNBestList_) {
      breakDowns.push_back(bestCosts);
  }

  std::map<size_t, size_t> batchMap;
  size_t tmp = 0;
  for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
    for (size_t t = 0; t < beamSizes[batchID]; ++t) {
      batchMap[tmp++] = batchID;
    }
  }

  for (size_t i = 0; i < beamSizeSum; i++) {
    size_t wordIndex = bestKeys[i] % Probs.dim(1);
    if (isInputFiltered_) {
      wordIndex = filterIndices[wordIndex];
    }

    size_t hypIndex  = bestKeys[i] / Probs.dim(1);
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

void BestHyps::CalcBeam(const Hypotheses& prevHyps,
                        const std::vector<ScorerPtr>& scorers,
                        const Words& filterIndices,
                        Beams &beams,
                        std::vector<uint>& beamSizes)
{
  BEGIN_TIMER("CalcBeam");

  using namespace mblas;

  mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());

  HostVector<float> vCosts;
  for (auto& h : prevHyps) {
    vCosts.push_back(h->GetCost());
  }
  mblas::copy(vCosts.begin(), vCosts.end(), Costs.begin());

  const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

  BroadcastVecColumn(weights_.at(scorers[0]->GetName()) * _1 + _2, Probs, Costs);

  for (size_t i = 1; i < scorers.size(); ++i) {
    mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

    Element(_1 + weights_.at(scorers[i]->GetName()) * _2, Probs, currProbs);
  }

  if (forbidUNK_) {
    DisAllowUNK(Probs);
  }

  size_t beamSizeSum = std::accumulate(beamSizes.begin(), beamSizes.end(), 0);

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;

  FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst);

  std::vector<HostVector<float>> breakDowns;
  if (returnNBestList_) {
      breakDowns.push_back(bestCosts);
      for (size_t i = 1; i < scorers.size(); ++i) {
        std::vector<float> modelCosts(beamSizeSum);
        mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

        nthElement_.getValueByKey(modelCosts, currProbs);
        breakDowns.push_back(modelCosts);
      }
  }

  std::map<size_t, size_t> batchMap;
  size_t tmp = 0;
  for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
    for (size_t t = 0; t < beamSizes[batchID]; ++t) {
      batchMap[tmp++] = batchID;
    }
  }

  for (size_t i = 0; i < beamSizeSum; i++) {
    size_t wordIndex = bestKeys[i] % Probs.dim(1);
    if (isInputFiltered_) {
      wordIndex = filterIndices[wordIndex];
    }

    size_t hypIndex  = bestKeys[i] / Probs.dim(1);
    float cost = bestCosts[i];

    HypothesisPtr prevHyp = prevHyps.at(hypIndex);
    HypothesisPtr hyp;
    if (returnAttentionWeights_) {
      hyp.reset(new Hypothesis(prevHyp, wordIndex, hypIndex, cost,
                               GetAlignments(scorers, hypIndex)));
    } else {
      hyp.reset(new Hypothesis(prevHyp, wordIndex, hypIndex, cost));
    }

    if(returnNBestList_) {
      hyp->GetCostBreakdown().resize(scorers.size());
      float sum = 0;
      for (size_t j = 0; j < scorers.size(); ++j) {
        if (j == 0)
          hyp->GetCostBreakdown()[0] = breakDowns[0][i];
        else {
          float cost = 0;
          if (j < scorers.size()) {
              if (prevHyps.at(hypIndex)->GetCostBreakdown().size() < scorers.size())
                const_cast<HypothesisPtr&>(prevHyps.at(hypIndex))->GetCostBreakdown().resize(scorers.size(), 0.0f);
              cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps.at(hypIndex))->GetCostBreakdown()[j];
          }
          sum += weights_.at(scorers[j]->GetName()) * cost;
          hyp->GetCostBreakdown()[j] = cost;
        }
      }
      hyp->GetCostBreakdown()[0] -= sum;
      hyp->GetCostBreakdown()[0] /= weights_.at(scorers[0]->GetName());
    }

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


