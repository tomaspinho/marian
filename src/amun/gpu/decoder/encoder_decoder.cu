// -*- mode: c++; tab-width: 2; indent-tabs-mode: nil -*-
#include <iostream>

#include "common/god.h"
#include "common/sentences.h"
#include "common/search.h"
#include "common/histories.h"

#include "encoder_decoder.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/decoder/enc_out_gpu.h"
#include "gpu/dl4mt/dl4mt.h"
#include "gpu/decoder/encoder_decoder_state.h"
#include "gpu/decoder/best_hyps.h"
#include "gpu/decoder/beam_size_gpu.h"

using namespace std;

namespace amunmt {
namespace GPU {

///////////////////////////////////////////////////////////////////////////////
EncoderDecoder::EncoderDecoder(
        const God &god,
        const std::string& name,
        const YAML::Node& config,
        size_t tab,
        const Weights& model,
        const Search &search)
: Scorer(god, name, config, tab, search),
  model_(model),
  encoder_(new Encoder(model_)),
  decoder_(new Decoder(god, model_)),
  indices_(god.Get<size_t>("beam-size")),
  encDecBuffer_(3)

{
  std::thread *thread = new std::thread( [&]{ DecodeAsync(god); });
  decThread_.reset(thread);

}

EncoderDecoder::~EncoderDecoder()
{
  decThread_->join();
}

State* EncoderDecoder::NewState() const {
  return new EDState();
}

void EncoderDecoder::Encode(const SentencesPtr source) {
  BEGIN_TIMER("Encode");

  EncOutPtr encOut(new EncOutGPU(source));

  if (source->size()) {
    encoder_->Encode(*source, tab_, encOut);
  }

  encDecBuffer_.Add(encOut);
  //cerr << "Encode encOut->sourceContext_=" << encOut->sourceContext_.Debug(0) << endl;

  PAUSE_TIMER("Encode");
}

void EncoderDecoder::BeginSentenceState(mblas::Matrix &states,
                                        mblas::Matrix &embeddings,
                                        BeamSizeGPU& beamSizes,
                                        size_t batchSize,
                                        const EncOut &encOut) const
{
  //cerr << "BeginSentenceState encOut->sourceContext_=" << encOut->sourceContext_.Debug(0) << endl;
  //cerr << "BeginSentenceState encOut->sentencesMask_=" << encOut->sentencesMask_.Debug(0) << endl;
  //cerr << "batchSize=" << batchSize << endl;

  decoder_->EmptyState(states, beamSizes, encOut, batchSize);

  decoder_->EmptyEmbedding(embeddings, batchSize);
}

void EncoderDecoder::Decode(const EDState& in,
                            mblas::Matrix &nextStateMatrix,
                            const BeamSizeGPU& beamSizes)
{
  BEGIN_TIMER("Decode");
  decoder_->Decode(nextStateMatrix,
                  in.GetStates(),
                  in.GetEmbeddings(),
                  beamSizes);
  PAUSE_TIMER("Decode");
}


void EncoderDecoder::DecodeAsync(const God &god)
{
  //cerr << "BeginSentenceState encOut->sourceContext_=" << encOut->sourceContext_.Debug(0) << endl;
  try {
    DecodeAsyncInternal(god);
  }
  catch(thrust::system_error &e)
  {
    std::cerr << "CUDA error during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(std::bad_alloc &e)
  {
    std::cerr << "Bad memory allocation during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(std::runtime_error &e)
  {
    std::cerr << "Runtime error during some_function: " << e.what() << std::endl;
    abort();
  }
  catch(...)
  {
    std::cerr << "Some other kind of error during some_function" << std::endl;
    abort();
  }
}

void EncoderDecoder::DecodeAsyncInternal(const God &god)
{
  boost::timer::cpu_timer timer;

  uint maxBeamSize = god.Get<uint>("beam-size");

  EDState state;
  mblas::Matrix nextStateMatrix;

  Hypotheses prevHyps;
  Histories histories(new BeamSizeGPU(), search_.NormalizeScore());
  size_t decoderStep;

  while (true) {
    if (histories.size() == 0) {
      // clean up previous
      CleanUpAfterSentence();

      LOG(progress)->info("Decoding took {}", timer.format(3, "%ws"));

      // read in next batch
      EncOutPtr encOut = encDecBuffer_.Get();
      assert(encOut);

      if (encOut->GetSentences().size() == 0) {
        break;
      }

      timer.start();

      // init states & histories/beams
      mblas::Matrix &bufStates = encOut->GetStates<mblas::Matrix>();
      mblas::Matrix &bufEmbeddings = encOut->GetEmbeddings<mblas::Matrix>();

      BeginSentenceState(bufStates,
                        bufEmbeddings,
                        static_cast<BeamSizeGPU&>(histories.GetBeamSizes()),
                        encOut->GetSentences().size(),
                        *encOut);

      mblas::Matrix &states = state.GetStates();
      mblas::Matrix &embeddings = state.GetEmbeddings();
      states.Copy(bufStates);
      embeddings.Copy(bufEmbeddings);

      histories.Init(maxBeamSize, encOut);
      prevHyps = histories.GetFirstHyps();

      decoderStep = 0;
    }

    //cerr << "beamSizes1=" << histories.GetBeamSizes().Debug(2) << endl;

    // decode
    boost::timer::cpu_timer timerStep;

    cerr << "1 state=" << state.Debug(1) << endl;
    cerr << "1 nextState=" << nextStateMatrix.Debug(1) << endl;

    //cerr << "beamSizes2=" << beamSizes.Debug(2) << endl;
    const BeamSizeGPU &bsGPU = static_cast<const BeamSizeGPU&>(histories.GetBeamSizes());
    Decode(state, nextStateMatrix, bsGPU);

    cerr << "2 state=" << state.Debug(1) << endl;
    cerr << "2 nextState=" << nextStateMatrix.Debug(1) << endl;

    //cerr << "beamSizes3=" << histories.GetBeamSizes().Debug(2) << endl;
    //cerr << "state=" << state->Debug(0) << endl;

    // beams
    if (decoderStep == 0) {
      histories.SetBeamSize(search_.MaxBeamSize());
    }
    //cerr << "beamSizes4=" << beamSizes.Debug(2) << endl;

    Beams beams;
    search_.BestHyps()->CalcBeam(prevHyps, *this, search_.FilterIndices(), beams, histories.GetBeamSizes());

    std::pair<Hypotheses, std::vector<uint> > histOut = histories.AddAndOutput(god, beams);
    Hypotheses &survivors = histOut.first;
    //const std::vector<uint> &completed = histOut.second;

    AssembleBeamState(nextStateMatrix, survivors, state);

    cerr << "3 state=" << state.Debug(1) << endl;
    cerr << "3 nextState=" << nextStateMatrix.Debug(1) << endl;

    /*
    cerr << "completed=" << Debug(completed, 2) << endl;
    cerr << "beamSizes=" << Debug(beamSizes, 2) << endl;
    cerr << "survivors=" << survivors.size() << endl;
    cerr << "beams=" << beams.size() << endl;
    cerr << "state=" << state->Debug(0) << endl;
    cerr << "nextState=" << nextState->Debug(0) << endl;
    cerr << "beamSizes5=" << histories.GetBeamSizes().Debug(2) << endl;
    cerr << "histories=" << histories.size() << endl;
    cerr << endl;
    */

    prevHyps.swap(survivors);
    ++decoderStep;

    LOG(progress)->info("Step took {}", timerStep.format(3, "%ws"));
  }
}


void EncoderDecoder::AssembleBeamState(const mblas::Matrix &nextStateMatrix,
                                const Hypotheses& hypos,
                                EDState& out)
{
  if (hypos.size() == 0) {
    return;
  }

  std::vector<size_t> beamWords;
  std::vector<uint> beamStateIds;
  for (const HypothesisPtr &h : hypos) {
     beamWords.push_back(h->GetWord());
     beamStateIds.push_back(h->GetPrevStateIndex());
  }
  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  //cerr << "beamStateIds=" << Debug(beamStateIds, 2) << endl;

  indices_.resize(beamStateIds.size());
  HostVector<uint> tmp = beamStateIds;

  mblas::copy(thrust::raw_pointer_cast(tmp.data()),
      beamStateIds.size(),
      thrust::raw_pointer_cast(indices_.data()),
      cudaMemcpyHostToDevice);
  //cerr << "indices_=" << mblas::Debug(indices_, 2) << endl;

  mblas::Assemble(out.GetStates(), nextStateMatrix, indices_);
  //cerr << "edOut.GetStates()=" << edOut.GetStates().Debug(1) << endl;

  //cerr << "beamWords=" << Debug(beamWords, 2) << endl;
  decoder_->Lookup(out.GetEmbeddings(), beamWords);
  //cerr << "edOut.GetEmbeddings()=" << edOut.GetEmbeddings().Debug(1) << endl;
}

void EncoderDecoder::GetAttention(mblas::Matrix& Attention) {
  decoder_->GetAttention(Attention);
}

BaseMatrix& EncoderDecoder::GetProbs() {
  return decoder_->GetProbs();
}

mblas::Matrix& EncoderDecoder::GetAttention() {
  return decoder_->GetAttention();
}

size_t EncoderDecoder::GetVocabSize() const {
  return decoder_->GetVocabSize();
}

void EncoderDecoder::Filter(const std::vector<size_t>& filterIds) {
  decoder_->Filter(filterIds);
}


}
}

